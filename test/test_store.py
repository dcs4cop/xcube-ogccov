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
from xcube.webapi.ows.coverages.request import CoverageRequest

from xcube.core.store import DataStoreError
from xcube_ogccov.store import OGCCovDataOpener, OGCCovDataStore


class Config:
    def __init__(self):
        self.server_root = "http://mock.com"
        self.coll_name = "collection1"
        self.schema_url = "http://mock2.com/some_schema_url"
        self.cov_url = (
            f"{self.server_root}/collections/"
            f"{self.coll_name}/coverage?f=netcdf"
        )
        self.coll_object = {
            "id": self.coll_name,
            "links": [
                {
                    "href": self.cov_url,
                    "rel": "http://www.opengis.net/def/rel/ogc/1.0/coverage",
                    "type": "application/x-netcdf",
                },
                {
                    "href": self.schema_url,
                    "rel": "http://www.opengis.net/def/rel/ogc/1.0/schema",
                    "type": "application/json",
                },
            ],
        }
        self.props = {
            f"prop{i}": {"type": "number", "x-ogc-property-seq": i}
            for i in range(1, 4)
        }

    def configure_mock(self, mock):
        # requests_mock.mock.case_sensitive = True
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
            self.cov_url,
            content=self.create_coverage
        )

    def create_coverage(self, request, context) -> bytes:
        cr = CoverageRequest(request.qs)
        ds = xr.Dataset(
            data_vars={
                prop: (["lat", "lon", "time"], np.zeros((3, 3, 3)))
                for prop in cr.properties
            },
            coords=dict(
                lon=[1, 2, 3],
                lat=[1, 2, 3],
                time=[1, 2, 3]
            ),
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
    schema = opener.get_open_data_params_schema('collection1').to_dict()
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


def test_open_data_with_error(requests_mock):
    CONFIG.configure_mock(requests_mock)
    requests_mock.get(
        CONFIG.cov_url,
        status_code=(status := 400),
        json={'error': {"message": (message := "oh no!")}},
    )
    opener = OGCCovDataOpener(server_url=CONFIG.server_root)
    with pytest.raises(DataStoreError) as e:
        opener.open_data(data_id=CONFIG.coll_name)
        assert str(status) in str(e)
        assert message in str(e)


def test_open_data(requests_mock):
    CONFIG.configure_mock(requests_mock)
    store = OGCCovDataStore(server_url=CONFIG.server_root)
    props = list(CONFIG.props.keys())[:2]
    ds = store.open_data(
        data_id=CONFIG.coll_name,
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
    assert props == list(ds.data_vars)
    # TODO: test other characteristics of the datacube
