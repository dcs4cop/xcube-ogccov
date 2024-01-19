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

import atexit
import os
import re
import shutil
import tempfile
import datetime
import urllib.parse
from typing import Any, Container, Collection
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import Union
import collections.abc

import requests
import xarray as xr
import xcube.core.normalize
from xcube.core.store import DATASET_TYPE
from xcube.core.store import DataDescriptor
from xcube.core.store import DataOpener
from xcube.core.store import DataStore
from xcube.core.store import DataStoreError
from xcube.core.store import DataTypeLike
from xcube.core.store import DatasetDescriptor
from xcube.core.store import DefaultSearchMixin
from xcube.util.jsonschema import (
    JsonBooleanSchema,
    JsonArraySchema,
    JsonNumberSchema,
)
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.undefined import UNDEFINED
from xcube_ogccov.constants import OGCCOV_DATA_OPENER_ID


class OGCCovDataOpener(DataOpener):
    """A data opener for OGC API - Coverages"""

    def __init__(self, server_url=None, normalize_names=True):
        """Instantiate an OGC API - Coverages data opener.

        :param server_url: URL of the API server
        """

        self._server_url = server_url
        self._normalize_names = normalize_names
        self._create_temporary_directory()

    def get_open_data_params_schema(
        self, data_id: Optional[str] = None
    ) -> JsonObjectSchema:
        self._assert_valid_data_id(data_id, allow_none=True)
        params = dict(
            subset=JsonObjectSchema(),  # TODO make the subset schema stricter
            bbox=JsonArraySchema(
                items=(
                    JsonNumberSchema(),
                    JsonNumberSchema(),
                    JsonNumberSchema(),
                    JsonNumberSchema(),
                ),
                description="bounding box (min_x, min_y, max_x, max_y)",
            ),
            datetime=JsonArraySchema(),
            properties=JsonArraySchema(
                items=(
                    JsonStringSchema(
                        enum=self._get_collection_properties(data_id)
                    )
                ),
            ),
            scale_factor=JsonNumberSchema(
                description="downscaling factor, applied on each axis"
            ),
            scale_axes=JsonObjectSchema(
                description="mapping from axis name to downscaling factor"
            ),
            scale_size=JsonObjectSchema(
                description="mapping from axis name to desired size"
            ),
            subset_crs=JsonStringSchema(
                description="CRS for the specified subset"
            ),
            bbox_crs=JsonStringSchema(
                description="CRS for the specified bbox"
            ),
            crs=JsonStringSchema(
                description="reproject the output to this CRS"
            ),
        )
        return JsonObjectSchema(
            properties=params, required=[], additional_properties=False
        )

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        # Unofficial parameters for testing, debugging, etc.
        # They're not in the schema so we remove them before validating.
        read_file_from = open_params.pop("_read_file_from", None)
        save_file_to = open_params.pop("_save_file_to", None)
        save_zarr_to = open_params.pop("_save_zarr_to", None)
        save_request_to = open_params.pop("_save_request_to", None)

        schema = self.get_open_data_params_schema(data_id)
        schema.validate_instance(open_params)

        # Fill in defaults from the schema
        props = schema.properties
        all_open_params = {
            k: props[k].default for k in props if props[k].default != UNDEFINED
        }
        all_open_params.update(open_params)

        ogc_params = [
            self.convert_store_param(p) for p in all_open_params.items()
        ]

        url = self._get_coverage_link(data_id)
        if "f" not in urllib.parse.parse_qs(urllib.parse.urlparse(url).query):
            ogc_params += [("f", "netcdf")]

        response = requests.get(url, params=dict(ogc_params))

        if response.status_code == 200:
            temp_subdir = tempfile.mkdtemp(dir=self._tempdir)
            filepath = os.path.join(temp_subdir, "dataset.nc")
            with open(filepath, "bw") as fh:
                fh.write(response.content)
            dataset = self._normalize_dataset(
                xr.open_dataset(filepath, engine="netcdf4")
            )

            if save_zarr_to:
                dataset.to_zarr(save_zarr_to)
            return dataset
        else:
            r = response.json()
            if "error" in r:
                e = r["error"]
                message = e["message"]
            else:
                message = response.content
            raise DataStoreError(
                f"Error opening data: {response.status_code}: {message}"
            )

    def get_data_ids(
        self,
        data_type: DataTypeLike = None,
        include_attrs: Container[str] = None,
    ) -> Union[Iterator[str], Iterator[Tuple[str, Dict[str, Any]]]]:
        response = requests.get(f"{self._server_url}/collections")
        collections = response.json()["collections"]
        return (collection["id"] for collection in collections)

    def has_data(self, data_id: str, data_type: Optional[str] = None) -> bool:
        return data_id in self.get_data_ids(data_type)

    @staticmethod
    def convert_store_param(kvp: Tuple[str, Any]) -> Tuple[str, str]:
        key, value = kvp
        if key in {"scale_factor", "subset_crs", "bbox_crs", "crs"}:
            # Pass through, converting underscores to hyphens if present
            return key.replace("_", "-"), value
        elif key == "datetime":
            if len(value) == 1:
                return "datetime", value
            elif len(value) == 2:
                return "datetime", "/".join(value)
            else:
                raise ValueError(f'Invalid datetime: "{value}"')
        elif key == "subset":
            return "subset", OGCCovDataOpener._subset_dict_to_string(value)
        elif key == "bbox":
            x0, y0, x1, y1 = value
            return "bbox", f"{x0},{y0},{x1},{y1}"
        elif key == "properties":
            return "properties", ",".join(value)
        elif key in {"scale_axes", "scale_size"}:
            return (
                key.replace("_", "-"),
                ",".join([f"{ax}({v})" for ax, v in value.items()]),
            )
        else:
            # This is a "can't happen", since validation against the schema
            # catches unknown parameters and this function matches all known
            # parameters. But if that accidentally changes in the future,
            # this is a good place to catch it.
            raise ValueError(f'Unknown parameter "{key}"')

    @staticmethod
    def _subset_dict_to_string(subset_dict: dict[str, Any]) -> str:
        parts = []
        for axis, range_ in subset_dict.items():
            if isinstance(range_, collections.abc.Sequence) and not isinstance(
                range_, str
            ):
                if len(range_) == 1:
                    parts.append(f"{axis}({range_[0]})")
                elif len(range_) == 2:
                    range_string = ":".join(
                        ["*" if x is None else f"{x}" for x in range_]
                    )
                    parts.append(f"{axis}({range_string})")
                else:
                    raise ValueError(
                        f"Invalid subset range {range_} for axis {axis}"
                    )
            else:
                parts.append(f"{axis}({str(range_)})")
        return ",".join(parts)

    def _normalize_dataset(self, dataset):
        dataset = xcube.core.normalize.normalize_dataset(dataset)

        dataset.coords["time"].attrs["standard_name"] = "time"
        if "lat" in dataset.coords:
            dataset.coords["lat"].attrs["standard_name"] = "latitude"
            dataset.coords["lat"].attrs["units"] = "degrees_north"
        if "lon" in dataset.coords:
            dataset.coords["lon"].attrs["standard_name"] = "longitude"
            dataset.coords["lon"].attrs["units"] = "degrees_east"

        if self._normalize_names:
            rename_dict = {}
            for name in dataset.data_vars.keys():
                normalized_name = re.sub(r"\W|^(?=\d)", "_", str(name))
                if name != normalized_name:
                    rename_dict[name] = normalized_name
            dataset_renamed = dataset.rename_vars(rename_dict)
            return dataset_renamed
        else:
            return dataset

    def _assert_valid_data_id(
        self, data_id: str, allow_none: bool = False
    ) -> None:
        if (data_id is None and not allow_none) or not self.has_data(data_id):
            raise ValueError(f'Unknown data id "{data_id}"')

    def _create_temporary_directory(self):
        # Create a temporary directory to hold downloaded files and a hook to
        # delete it when the interpreter exits. xarray.open reads data lazily
        # so we can't just delete the file after returning the Dataset. We
        # could also use weakref hooks to delete individual files when the
        # corresponding object is garbage collected, but even then the
        # directory is useful to group the files and offer an extra assurance
        # that they will be deleted.
        tempdir = tempfile.mkdtemp()

        def delete_tempdir():
            # This method is hard to unit test, so we exclude it from test
            # coverage reports.
            shutil.rmtree(tempdir, ignore_errors=True)  # pragma: no cover

        atexit.register(delete_tempdir)
        self._tempdir = tempdir

    def _get_coverage_link(self, collection_id):
        links = self._get_collection_links(
            collection_id,
            {
                "rel": {
                    "coverage",
                    "http://www.opengis.net/def/rel/ogc/1.0/coverage",
                },
                "type": {
                    "netcdf",
                    "application/netcdf",
                    "application/x-netcdf",
                },
            },
        )
        if len(links) > 0:
            # If multiple coverage links available, use the first.
            return links[0]
        else:
            # Fall back to standard endpoint if none specified explicitly.
            return f"{self._server_url}/collections/{collection_id}/coverage"

    def _get_collection_properties(self, collection_id: str):
        url = self._get_collection_link(
            collection_id,
            {
                "rel": {"http://www.opengis.net/def/rel/ogc/1.0/schema"},
                "type": {"application/json"},
            },
            f"collections/{collection_id}/schema",
        )
        response = requests.get(url)
        schema = response.json()
        return list(schema["properties"].keys())

    def _get_collection_link(
        self,
        collection_id: str,
        selectors: dict[str, Collection[str]],
        fallback: str,
    ) -> str:
        links = self._get_collection_links(collection_id, selectors)
        if len(links) > 0:
            # If multiple collection links available, use the first.
            return links[0]
        else:
            # Use supplied fallback link if none found in collection links
            return self._server_url + "/" + fallback

    def _get_collection_links(
        self, collection_id: str, selectors: dict[str, Collection[str]]
    ) -> list[str]:
        response = requests.get(
            f"{self._server_url}/collections/{collection_id}"
        )
        collection = response.json()
        result = []
        for link in collection.get("links", []):
            if (
                all([link.get(prop) in selectors[prop] for prop in selectors])
                and "href" in link
            ):
                result.append(link["href"])
        return result


class OGCCovDataStore(DefaultSearchMixin, OGCCovDataOpener, DataStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_data_store_params_schema(cls) -> JsonObjectSchema:
        params = dict(normalize_names=JsonBooleanSchema(default=False))

        store_params = dict(
            server_url=JsonStringSchema(),
        )

        params.update(store_params)
        return JsonObjectSchema(
            properties=params, required=None, additional_properties=False
        )

    @classmethod
    def get_data_types(cls) -> Tuple[str, ...]:
        return (DATASET_TYPE.alias,)

    def get_data_types_for_data(self, data_id: str) -> Tuple[str, ...]:
        self._assert_valid_data_id(data_id)
        return (DATASET_TYPE.alias,)

    def describe_data(
        self, data_id: str, data_type: Optional[str] = None
    ) -> DatasetDescriptor:
        self._assert_valid_data_id(data_id)
        self._validate_data_type(data_type)

        return DatasetDescriptor(
            data_id=data_id,
            data_type=DATASET_TYPE,
            time_period=None,  # str
            spatial_res=None,  # float
            coords=None,  # Mapping[str, 'VariableDescriptor'],
            data_vars=None,  # Mapping[str, 'VariableDescriptor'],
            attrs=None,  # Mapping[Hashable, any],
            open_params_schema=None,  # JsonObjectSchema,
            **(self._domainset_params(data_id)),
        )

    def _domainset_params(self, data_id: str) -> dict[str, Any]:
        params = {}
        domainset = requests.get(
            f"{self._server_url}/collections/{data_id}/coverage/domainset",
            params=dict(f="json"),
        ).json()
        grid = domainset.get("generalGrid", {})
        params["dims"] = {}
        bbox = [None, None, None, None]
        for index, axis in enumerate(grid.get("axis", [])):
            label = axis.get("axisLabel", "?")
            params["dims"][label] = index
            if label in {"lon", "longitude", "x"}:
                bbox[0] = float(axis.get("lowerBound", "0"))
                bbox[2] = float(axis.get("upperBound", "0"))
            elif label in {"lat", "latitude", "y"}:
                bbox[1] = float(axis.get("lowerBound", "0"))
                bbox[3] = float(axis.get("upperBound", "0"))
            elif label in {"time", "t"}:
                try:
                    params["time_range"] = tuple(
                        datetime.datetime.fromisoformat(
                            axis.get(bound)
                        ).strftime("%Y-%m-%d")
                        for bound in ["lowerBound", "upperBound"]
                    )
                except ValueError:
                    # Ignore malformed timestamps
                    pass
        params["bbox"] = None if None in bbox else tuple(bbox)
        params["crs"] = grid.get("srsName")
        return params

    # noinspection PyTypeChecker
    def search_data(
        self, data_type: Optional[DataTypeLike] = None, **search_params
    ) -> Iterator[DataDescriptor]:
        self._validate_data_type(data_type)
        return super().search_data(data_type=data_type, **search_params)

    def get_data_opener_ids(
        self, data_id: Optional[str] = None, data_type: Optional[str] = None
    ) -> Tuple[str, ...]:
        self._validate_data_type(data_type)
        self._assert_valid_data_id(data_id, allow_none=True)
        return (OGCCOV_DATA_OPENER_ID,)

    def get_open_data_params_schema(
        self, data_id: Optional[str] = None, opener_id: Optional[str] = None
    ) -> JsonObjectSchema:
        # At present, there's only one opener ID available, so we do nothing
        # with it except to check that it was correct (or None).
        self._assert_valid_opener_id(opener_id)
        self._assert_valid_data_id(data_id, allow_none=True)
        return super().get_open_data_params_schema(data_id)

    def open_data(
        self, data_id: str, opener_id: Optional[str] = None, **open_params
    ) -> xr.Dataset:
        self._assert_valid_opener_id(opener_id)
        self._assert_valid_data_id(data_id)
        return super().open_data(data_id, **open_params)

    ###########################################################################
    # Implementation helpers

    @staticmethod
    def _validate_data_type(data_type: DataTypeLike):
        if not OGCCovDataStore._is_data_type_satisfied(data_type):
            raise DataStoreError(
                f"Supplied data type {data_type!r} is not compatible"
                f' with "{DATASET_TYPE!r}."'
            )

    @staticmethod
    def _is_data_type_satisfied(data_type: DataTypeLike) -> bool:
        # We expect all datasets to be available as cubes, so we simply check
        # against TYPE_SPECIFIER_CUBE.
        if data_type is None:
            return True
        return DATASET_TYPE.is_super_type_of(data_type)

    @staticmethod
    def _assert_valid_opener_id(opener_id):
        if opener_id is not None and opener_id != OGCCOV_DATA_OPENER_ID:
            raise DataStoreError(
                f'Data opener identifier must be "{OGCCOV_DATA_OPENER_ID}"'
                f'but got "{opener_id}"'
            )
