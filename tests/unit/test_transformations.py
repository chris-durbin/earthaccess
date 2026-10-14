import datetime
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import earthaccess
import pytest
from earthaccess.exceptions import TransformationFailed, TransformationNotSupported
from earthaccess.results import DataGranule
from earthaccess.system import PROD, UAT
from earthaccess.transformations import (
    TransformationBackend,
    TransformationRequest,
    available_backends,
    get_backend,
    register_backend,
    registry,
)
from earthaccess.transformations.base import normalize_temporal

harmony = pytest.importorskip("harmony")

from earthaccess.transformations.harmony import HarmonyBackend  # noqa: E402
from harmony.client import ProcessingFailedException  # noqa: E402


def naive(*args: int) -> datetime.datetime:
    # normalize_temporal produces naive datetimes in UTC.
    return datetime.datetime(*args)  # noqa: DTZ001


TOKEN = "TOKEN"  # noqa: S105

CAPABILITIES = {
    "conceptId": "C1-PROV",
    "shortName": "EXAMPLE",
    "variableSubset": True,
    "bboxSubset": True,
    "shapeSubset": False,
    "temporalSubset": True,
    "concatenate": False,
    "reproject": False,
    "outputFormats": ["application/x-netcdf4"],
    "services": [{"name": "example-service"}],
    "variables": [{"name": "/sst", "conceptId": "V1-PROV"}, {"name": "ice"}],
}


def make_granule(concept_id: str, collection_id: str) -> DataGranule:
    return DataGranule(
        {
            "meta": {
                "concept-id": concept_id,
                "collection-concept-id": collection_id,
            },
            "umm": {},
        },
    )


@pytest.fixture
def auth():
    auth = earthaccess.Auth()
    auth.authenticated = True
    auth.token = {"access_token": TOKEN}
    return auth


@pytest.fixture
def backend(auth):
    backend = HarmonyBackend(auth)
    backend.client = MagicMock()
    return backend


@pytest.fixture
def logged_in(auth, monkeypatch):
    monkeypatch.setattr(earthaccess, "_auth", auth)
    return auth


class FakeBackend(TransformationBackend):
    name = "fake"
    calls: ClassVar[list[tuple[str, Any]]] = []

    def capabilities(self, collection_id):
        self.calls.append(("capabilities", collection_id))
        return {}

    def validate(self, request, capabilities):  # noqa: ARG002
        self.calls.append(("validate", request))

    def submit(self, request):
        self.calls.append(("submit", request))
        return f"job-{request.collection_id}"

    def wait(self, job_id, *, show_progress):  # noqa: ARG002
        self.calls.append(("wait", job_id))
        return "successful", "done"

    def result_urls(self, job_id):
        return [f"https://example.com/{job_id}/out.nc"]

    def download(self, urls, local_path, *, force):  # noqa: ARG002
        self.calls.append(("download", local_path))
        return [local_path / Path(url).name for url in urls]


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setitem(registry._backends, FakeBackend.name, FakeBackend)
    FakeBackend.calls = []
    return FakeBackend


class TestRegistry:
    def test_harmony_is_registered(self):
        assert "harmony" in available_backends()
        assert get_backend("harmony") is HarmonyBackend

    def test_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown transformation backend"):
            get_backend("nope")

    def test_register_duplicate(self, fake_backend):
        with pytest.raises(ValueError, match="already registered"):
            register_backend(fake_backend)
        register_backend(fake_backend, replace=True)


class TestNormalizeTemporal:
    def test_strings(self):
        assert normalize_temporal(("2024-01-01", "2024-01-02T12:00:00Z")) == (
            naive(2024, 1, 1),
            naive(2024, 1, 2, 12),
        )

    def test_open_ended_date_only_stop_includes_whole_day(self):
        assert normalize_temporal((None, "2024-01-02")) == (
            None,
            naive(2024, 1, 2, 23, 59, 59),
        )

    def test_timezone_converted_to_utc(self):
        start, _ = normalize_temporal(("2024-01-01T02:00:00+02:00", None))
        assert start == naive(2024, 1, 1)

    def test_reversed(self):
        with pytest.raises(ValueError, match="after"):
            normalize_temporal(("2024-01-02", "2024-01-01"))


class TestHarmonyBackend:
    def test_environment_follows_login_system(self, auth):
        assert (
            HarmonyBackend(auth).client.config.environment == harmony.Environment.PROD
        )
        auth._set_earthdata_system(UAT)
        assert HarmonyBackend(auth).client.config.environment == harmony.Environment.UAT
        auth._set_earthdata_system(PROD)

    def test_client_uses_earthaccess_token(self, auth):
        assert HarmonyBackend(auth).client.token == TOKEN

    def test_capabilities(self, backend):
        backend.client.submit.return_value = CAPABILITIES
        assert backend.capabilities("C1-PROV") == CAPABILITIES
        request = backend.client.submit.call_args.args[0]
        assert isinstance(request, harmony.CapabilitiesRequest)
        assert request.collection_id == "C1-PROV"

    def test_validate_supported(self, backend):
        request = TransformationRequest(
            collection_id="C1-PROV",
            variables=["sst", "/ice", "V2-PROV"],
            bounding_box=(-10, -10, 10, 10),
            output_format="application/x-netcdf4",
        )
        backend.validate(request, CAPABILITIES)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"crs": "EPSG:3413"}, "reprojection"),
            ({"concatenate": True}, "concatenation"),
            ({"output_format": "image/png"}, "output format"),
            ({"variables": ["salinity"]}, "unknown variables: salinity"),
        ],
    )
    def test_validate_unsupported(self, backend, kwargs, match):
        request = TransformationRequest(collection_id="C1-PROV", **kwargs)
        with pytest.raises(TransformationNotSupported, match=match):
            backend.validate(request, CAPABILITIES)

    def test_validate_no_services(self, backend):
        request = TransformationRequest(collection_id="C1-PROV")
        with pytest.raises(TransformationNotSupported, match="no Harmony services"):
            backend.validate(request, {**CAPABILITIES, "services": []})

    def test_build_request(self, backend):
        request = backend.build_request(
            TransformationRequest(
                collection_id="C1-PROV",
                granule_ids=["G1-PROV"],
                variables=["sst"],
                bounding_box=(-10, -20, 10, 20),
                temporal=(naive(2024, 1, 1), None),
                output_format="application/x-netcdf4",
                options={"skip_preview": True},
            ),
        )
        assert request.collection.id == "C1-PROV"
        assert request.granule_id == ["G1-PROV"]
        assert request.variables == ["sst"]
        assert request.spatial == harmony.BBox(-10, -20, 10, 20)
        assert request.temporal == {"start": naive(2024, 1, 1)}
        assert request.format == "application/x-netcdf4"
        assert request.skip_preview is True
        assert "earthaccess" in request.labels

    def test_build_invalid_request(self, backend):
        with pytest.raises(ValueError, match="Invalid Harmony request"):
            backend.build_request(
                TransformationRequest(
                    collection_id="C1-PROV",
                    bounding_box=(-10, 20, 10, -20),
                ),
            )

    def test_submit_returns_job_id(self, backend):
        backend.client.submit.return_value = "job-1"
        assert backend.submit(TransformationRequest(collection_id="C1-PROV")) == "job-1"
        backend.client.submit.return_value = {"jobID": "job-2", "status": "successful"}
        assert backend.submit(TransformationRequest(collection_id="C1-PROV")) == "job-2"

    @pytest.mark.parametrize("status", ["successful", "complete_with_errors"])
    def test_wait_success(self, backend, status):
        backend.client.progress.return_value = (100, status, "ok")
        assert backend.wait("job-1", show_progress=False) == (status, "ok")

    @pytest.mark.parametrize("status", ["canceled", "paused"])
    def test_wait_incomplete(self, backend, status):
        backend.client.progress.return_value = (40, status, "stopped")
        with pytest.raises(TransformationFailed, match=status) as exc_info:
            backend.wait("job-1", show_progress=False)
        assert exc_info.value.job_id == "job-1"

    def test_wait_failed(self, backend):
        backend.client.wait_for_processing.side_effect = ProcessingFailedException(
            "job-1",
            "boom",
        )
        with pytest.raises(TransformationFailed, match="boom"):
            backend.wait("job-1", show_progress=False)

    def test_download_skips_zarr(self, backend, tmp_path):
        future = MagicMock()
        future.result.return_value = str(tmp_path / "out.nc")
        backend.client.download.return_value = future
        files = backend.download(
            ["https://h/out.nc", "s3://h/out.zarr"],
            tmp_path,
            force=False,
        )
        assert files == [tmp_path / "out.nc"]
        backend.client.download.assert_called_once_with(
            "https://h/out.nc",
            directory=str(tmp_path),
            overwrite=False,
        )


class TestTransform:
    @pytest.mark.usefixtures("fake_backend")
    def test_requires_login(self, monkeypatch):
        monkeypatch.setattr(earthaccess, "_auth", earthaccess.Auth())
        with pytest.raises(RuntimeError, match="login"):
            earthaccess.transform(collection="C1-PROV", backend="fake")

    @pytest.mark.usefixtures("logged_in", "fake_backend")
    def test_requires_granules_xor_collection(self):
        with pytest.raises(ValueError, match="Exactly one"):
            earthaccess.transform(backend="fake")

    @pytest.mark.usefixtures("logged_in")
    def test_one_job_per_collection(self, fake_backend, tmp_path):
        granules = [
            make_granule("G1-A", "C1-A"),
            make_granule("G2-B", "C2-B"),
            make_granule("G3-A", "C1-A"),
        ]
        results = earthaccess.transform(
            granules,
            variables=["sst"],
            temporal=("2024-01-01", "2024-01-02"),
            local_path=tmp_path,
            backend="fake",
            show_progress=False,
        )

        assert [r.job_id for r in results] == ["job-C1-A", "job-C2-B"]
        assert results[0].files == [tmp_path / "out.nc"]
        submitted = [arg for name, arg in fake_backend.calls if name == "submit"]
        assert submitted[0].granule_ids == ["G1-A", "G3-A"]
        assert submitted[0].temporal == (
            naive(2024, 1, 1),
            naive(2024, 1, 2, 23, 59, 59),
        )
        assert submitted[1].granule_ids == ["G2-B"]

    @pytest.mark.usefixtures("logged_in")
    def test_no_download(self, fake_backend):
        (result,) = earthaccess.transform(
            collection="C1-A",
            download=False,
            validate_capabilities=False,
            backend="fake",
            show_progress=False,
        )
        assert result.urls == ["https://example.com/job-C1-A/out.nc"]
        assert result.files == []
        names = [name for name, _ in fake_backend.calls]
        assert "capabilities" not in names
        assert "download" not in names

    @pytest.mark.usefixtures("logged_in")
    def test_transformation_capabilities(self, fake_backend):
        granule = make_granule("G1-A", "C1-A")
        assert earthaccess.transformation_capabilities(granule, backend="fake") == {}
        assert fake_backend.calls == [("capabilities", "C1-A")]
