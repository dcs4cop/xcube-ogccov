# MIT License
#
# Copyright (c) 2023–2024 Brockmann Consult GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tempfile
import os
from typing import Any

import pytest

import numpy as np
import xarray as xr
from xcube.webapi.ows.coverages.request import CoverageRequest

from xcube.core.store import DataStoreError
from xcube_ogccov.store import OGCCovDataOpener, OGCCovDataStore


class Config:
    def __init__(self, include_coverage_link=True, include_schema_link=True):
        self.server_root = "http://mock.com"
        self.coll_name = "collection1"
        self.schema_url = (
            "http://mock2.com/some_schema_url"
            if include_schema_link
            else f"{self.server_root}/collections/{self.coll_name}/schema"
        )
        self.cov_url = (
            f"{self.server_root}/collections/{self.coll_name}/coverage"
        )
        links = (
            [
                {
                    "href": self.cov_url,
                    "rel": "http://www.opengis.net/def/rel/ogc/1.0/coverage",
                    "type": "application/x-netcdf",
                }
            ]
            if include_coverage_link
            else []
        )
        links += (
            [
                {
                    "href": self.schema_url,
                    "rel": "http://www.opengis.net/def/rel/ogc/1.0/schema",
                    "type": "application/json",
                },
            ]
            if include_schema_link
            else []
        )
        self.coll_object = {
            "id": self.coll_name,
            "links": links,
        }
        self.props = {
            f"prop-{i}": {"type": "number", "x-ogc-property-seq": i}
            for i in range(1, 4)
        }

        def ax(label, lb, ub):
            return {
                "axisLabel": label,
                "lowerBound": lb,
                "resolution": 1,
                "type": "RegularAxis",
                "uomLabel": "unknown",
                "upperBound": ub,
            }

        self.domainset = {
            "generalGrid": {
                "axis": [
                    ax("lat", 30, 35),
                    ax("lon", 30, 35),
                    ax("time", "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"),
                ],
                "axisLabels": ["lat", "lon", "time"],
                "gridLimits": {
                    "axis": [
                        {
                            "axisLabel": "lat",
                            "lowerBound": 0,
                            "type": "IndexAxis",
                            "upperBound": 5,
                        },
                        {
                            "axisLabel": "lon",
                            "lowerBound": 0,
                            "type": "IndexAxis",
                            "upperBound": 5,
                        },
                    ],
                    "axisLabels": ["lat", "lon", "time"],
                    "srsName": "http://www.opengis.net/def/crs/OGC/0/Index2D",
                    "type": "GridLimits",
                },
                "srsName": "EPSG:4326",
                "type": "GeneralGridCoverage",
            },
            "type": "DomainSet",
        }

    def configure_mock(self, mock, coverage_link_includes_format=True):
        mock.get(
            self.server_root + "/collections",
            json={"collections": [self.coll_object]},
        )
        mock.get(
            f"{self.server_root}/collections/{self.coll_name}",
            json=self.coll_object,
        )
        mock.get(self.schema_url, json={"properties": self.props})
        mock.get(
            self.cov_url
            + ("?f=netcdf" if coverage_link_includes_format else ""),
            content=self.create_coverage,
        )
        mock.get(
            f"{self.server_root}/collections/"
            f"{self.coll_name}/coverage/domainset",
            json=self.domainset,
        )

    def create_coverage(self, request, context) -> bytes:
        cr = CoverageRequest(request.qs)
        ds = xr.Dataset(
            data_vars={
                prop: (["lat", "lon", "time"], np.zeros((3, 3, 3)))
                for prop in cr.properties
            },
            coords=dict(lon=[1, 2, 3], lat=[1, 2, 3], time=[1, 2, 3]),
            attrs=dict(description="Foo"),
        )
        with tempfile.TemporaryDirectory() as parent:
            path = os.path.join(parent, "temp.nc")
            ds.to_netcdf(path=path)
            with open(path, "br") as fh:
                content = fh.read()
                return content


CONFIG = Config()


def test_get_open_data_params_schema(requests_mock):
    CONFIG.configure_mock(requests_mock)
    opener = OGCCovDataOpener(server_url=CONFIG.server_root)
    schema = opener.get_open_data_params_schema("collection1").to_dict()
    assert {
        "subset",
        "bbox",
        "datetime",
        "properties",
        "scale_factor",
        "scale_axes",
        "scale_size",
        "subset_crs",
        "bbox_crs",
        "crs",
    } == set(schema["properties"].keys())
    assert set(schema["properties"]["properties"]["items"]["enum"]) == set(
        CONFIG.props.keys()
    )


def test_get_open_data_params_schema_invalid_data_id(requests_mock):
    CONFIG.configure_mock(requests_mock)
    opener = OGCCovDataOpener(server_url=CONFIG.server_root)
    with pytest.raises(ValueError) as e:
        opener.get_open_data_params_schema(data_id := "invalid_id").to_dict()
        assert data_id in str(e)
        assert "unknown" in str(e).lower()


@pytest.mark.parametrize("structured_error_response", [False, True])
def test_open_data_with_error(requests_mock, structured_error_response):
    CONFIG.configure_mock(requests_mock)
    message = "oh no!"
    requests_mock.get(
        url=CONFIG.cov_url,
        status_code=(status := 400),
        json={"error": {"message": message}}
        if structured_error_response
        else {"json": {"some_unknown_key": message}},
    )
    opener = OGCCovDataOpener(server_url=CONFIG.server_root)
    with pytest.raises(DataStoreError) as e:
        opener.open_data(data_id=CONFIG.coll_name)
        assert str(status) in str(e)
        assert message in str(e)


@pytest.mark.parametrize("include_coverage_link", [False, True])
@pytest.mark.parametrize("include_schema_link", [False, True])
@pytest.mark.parametrize("coverage_link_includes_format", [False, True])
@pytest.mark.parametrize("normalize_names", [False, True])
def test_open_data(
    requests_mock,
    coverage_link_includes_format,
    include_coverage_link,
    include_schema_link,
    normalize_names,
):
    config = Config(
        include_coverage_link=include_coverage_link,
        include_schema_link=include_schema_link,
    )
    config.configure_mock(requests_mock, coverage_link_includes_format)
    store = OGCCovDataStore(
        server_url=config.server_root, normalize_names=normalize_names
    )
    props = list(config.props.keys())[:2]
    ds = store.open_data(
        data_id=config.coll_name,
        subset=dict(lat=(30, 35), lon=(30, 35)),
        bbox=[30, 30, 35, 35],
        datetime=["2020-01-01T00:00:00Z"],
        properties=props,
        scale_factor=1,
        scale_axes=dict(lat=1, lon=1),
        scale_size=dict(lat=10, lon=10),
        subset_crs="EPSG:4326",
        bbox_crs="EPSG:4326",
        crs="EPSG:4326",
    )
    normalized_props = [p.replace("-", "_") for p in props]
    assert (normalized_props if normalize_names else props) == list(
        ds.data_vars
    )
    # TODO: test other characteristics of the datacube


def test_open_data_unknown_id(requests_mock):
    CONFIG.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=CONFIG.server_root)
    with pytest.raises(ValueError) as e:
        store.open_data(
            data_id=CONFIG.coll_name,
            subset=dict(lat=(1, 2, 3)),
        )
        assert "unknown data id" in str(e).lower()


def test_open_data_invalid_datetime(requests_mock):
    CONFIG.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=CONFIG.server_root)
    with pytest.raises(ValueError) as e:
        store.open_data(
            data_id=CONFIG.coll_name,
            datetime=["foo", "bar", "baz"],
        )
        assert "invalid datetime" in str(e).lower()


def test_open_data_invalid_subset(requests_mock):
    CONFIG.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=CONFIG.server_root)
    with pytest.raises(ValueError) as e:
        store.open_data(
            data_id=CONFIG.coll_name,
            subset=dict(lat=(1, 2, 3)),
        )
        assert "invalid subset" in str(e).lower()


def test_unknown_parameter(requests_mock):
    CONFIG.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=CONFIG.server_root)
    with pytest.raises(ValueError) as e:
        store.convert_store_param(
            (param := "invalid_parameter", "arbitrary value")
        )
        assert "unknown" in str(e).lower()
        assert param in str(e)


def test_get_data_store_params_schema():
    assert {
        "additionalProperties": False,
        "properties": {
            "normalize_names": {"default": False, "type": "boolean"},
            "server_url": {"type": "string"},
        },
        "type": "object",
    } == OGCCovDataStore.get_data_store_params_schema().to_dict()


def test_get_data_types():
    assert ("dataset",) == OGCCovDataStore.get_data_types()


def test_get_data_types_for_data(requests_mock):
    CONFIG.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=CONFIG.server_root)
    assert ("dataset",) == store.get_data_types_for_data(CONFIG.coll_name)


def test_get_data_opener_ids(requests_mock):
    CONFIG.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=CONFIG.server_root)
    assert ("dataset:netcdf:ogccov",) == store.get_data_opener_ids(
        CONFIG.coll_name
    )


def test_search_data(requests_mock):
    CONFIG.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=CONFIG.server_root)
    search_results = list(store.search_data())
    assert 1 == len(search_results)
    assert search_results[0].to_dict() == {
        "data_id": CONFIG.coll_name,
        "data_type": "dataset",
        "crs": "EPSG:4326",
        "bbox": [30.0, 30.0, 35.0, 35.0],
        "dims": {"lat": 0, "lon": 1, "time": 2},
        "time_range": ["2020-01-01", "2020-01-02"],
    }
