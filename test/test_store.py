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

import pytest

import numpy as np
import xarray as xr
import pandas as pd
from pyproj import CRS
from xcube.webapi.ows.coverages.request import CoverageRequest

from xcube.core.store import DataStoreError, GEO_DATA_FRAME_TYPE
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

        def grid(label, ub):
            return (
                {
                    "axisLabel": label,
                    "lowerBound": 0,
                    "type": "IndexAxis",
                    "upperBound": ub,
                },
            )

        self.domainset = {
            "generalGrid": {
                "axis": [
                    ax("lat", 30, 35),
                    ax("lon", 30, 35),
                    ax("time", "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"),
                ],
                "axisLabels": ["lat", "lon", "time"],
                "gridLimits": {
                    "axis": [grid("lat", 5), grid("lon", 5), grid("time", 1)],
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
        expected = {
            "subset": [["lat(50:55),lon(5:10)"]],
            "bbox": [["50,5,55,10"]],
            "datetime": [
                ["2020-01-01T00:00:00Z"],
                ["2020-01-01T00:00:00Z/2020-01-02T00:00:00Z"],
            ],
            "scale-factor": [["1"]],
            "scale-axes": [["lat(1),lon(1)"]],
            "scale-size": [["lat(6),lon(6)"]],
            "subset-crs": [["EPSG:4326"]],
            "bbox-crs": [["EPSG:4326"]],
            "crs": [["EPSG:4326"]],
            "f": [["netcdf"]],
        }
        cr = CoverageRequest(request.qs)

        assert all([request.qs[k] in v for k, v in expected.items()])
        ds = xr.Dataset(
            data_vars={
                prop: (
                    ["lat", "lon", "time"],
                    np.zeros((6, 6, 1 if isinstance(cr.datetime, str) else 2)),
                )
                for prop in cr.properties
            },
            coords=dict(
                lon=np.arange(5, 11),
                lat=np.arange(50, 56),
                time=[pd.Timestamp(cr.datetime).to_datetime64()]
                if isinstance(cr.datetime, str)
                else [
                    pd.Timestamp(cr.datetime[0]).to_datetime64(),
                    pd.Timestamp(cr.datetime[1]).to_datetime64(),
                ],
            ),
            attrs=dict(description="test dataset"),
        )
        ds.rio.write_crs("EPSG:4326", inplace=True)
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


def test_get_open_data_params_schema_invalid_opener_id(requests_mock):
    CONFIG.configure_mock(requests_mock)
    opener = OGCCovDataStore(server_url=CONFIG.server_root)
    with pytest.raises(DataStoreError) as e:
        opener.get_open_data_params_schema(
            opener_id=(opener_id := "invalid_id")
        ).to_dict()
        assert opener_id in str(e)
        assert "opener" in str(e).lower()


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
@pytest.mark.parametrize(
    "datetime_param",
    [
        ["2020-01-01T00:00:00Z"],
        ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
    ],
)
def test_open_data(
    requests_mock,
    coverage_link_includes_format,
    include_coverage_link,
    include_schema_link,
    normalize_names,
    datetime_param,
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
    bbox = [50, 5, 55, 10]
    ds = store.open_data(
        data_id=config.coll_name,
        subset=dict(lat=(bbox[0], bbox[2]), lon=(bbox[1], bbox[3])),
        bbox=bbox,
        datetime=datetime_param,
        properties=props,
        scale_factor=1,
        scale_axes=dict(lat=1, lon=1),
        scale_size=dict(lat=6, lon=6),
        subset_crs="EPSG:4326",
        bbox_crs="EPSG:4326",
        crs="EPSG:4326",
    )
    normalized_props = [p.replace("-", "_") for p in props]
    assert list(ds.data_vars) == (
        normalized_props if normalize_names else props
    )

    assert ds.rio.crs == CRS("EPSG:4326")
    assert np.array_equal(ds.lat, np.arange(bbox[0], bbox[2] + 1))
    assert np.array_equal(ds.lon, np.arange(bbox[1], bbox[3] + 1))
    assert [pd.Timestamp(t, tz="UTC") for t in ds.time.values] == [
        pd.Timestamp(t) for t in datetime_param
    ]


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


def test_open_data_save_zarr(requests_mock):
    CONFIG.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=CONFIG.server_root)
    with tempfile.TemporaryDirectory() as parent:
        path = os.path.join(parent, "test.zarr")
        bbox = [50, 5, 55, 10]
        store.open_data(
            data_id=CONFIG.coll_name,
            subset=dict(lat=(bbox[0], bbox[2]), lon=(bbox[1], bbox[3])),
            bbox=bbox,
            datetime=["2020-01-01T00:00:00Z"],
            properties=["prop-1"],
            scale_factor=1,
            scale_axes=dict(lat=1, lon=1),
            scale_size=dict(lat=6, lon=6),
            subset_crs="EPSG:4326",
            bbox_crs="EPSG:4326",
            crs="EPSG:4326",
            _save_zarr_to=path,
        )
        xr.open_zarr(path)


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


def test_describe_data_malformed_time(requests_mock):
    config = Config()
    config.domainset["generalGrid"]["axis"][2]["lowerBound"] = "malformed"
    config.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=config.server_root)
    assert store.describe_data(config.coll_name).to_dict() == {
        "data_id": CONFIG.coll_name,
        "data_type": "dataset",
        "crs": "EPSG:4326",
        "bbox": [30.0, 30.0, 35.0, 35.0],
        "dims": {"lat": 0, "lon": 1, "time": 2},
    }


def test_describe_data_invalid_data_type(requests_mock):
    CONFIG.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=CONFIG.server_root)
    with pytest.raises(DataStoreError) as e:
        store.describe_data(
            data_id=CONFIG.coll_name,
            data_type=(data_type := GEO_DATA_FRAME_TYPE),
        )
        assert str(data_type) in str(e)
        assert "not compatible" in str(e).lower()


def test_subset_dict_to_string():
    assert (
        OGCCovDataStore.subset_dict_to_string(dict(ax1=["val1"], ax2="val2"))
        == "ax1(val1),ax2(val2)"
    )
