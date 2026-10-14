"""Transformation backend for NASA Harmony, built on harmony-py.

Requires the `earthaccess[harmony]` optional extra.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from earthaccess.exceptions import TransformationFailed, TransformationNotSupported
from earthaccess.system import PROD, UAT

from .base import TransformationBackend, TransformationRequest

if TYPE_CHECKING:
    import harmony

    from earthaccess.auth import Auth

logger = logging.getLogger(__name__)

DEFAULT_LABEL = "earthaccess"

# Statuses Harmony jobs may end in without having failed outright.
SUCCESSFUL_STATUSES = {"successful", "complete_with_errors"}

# Variable values Harmony accepts that are not variable names.
_ALL_VARIABLES = "all"
_VARIABLE_CONCEPT_ID = re.compile(r"^V\d+-[A-Z0-9_]+$")

# Capabilities flags that must be set for each `TransformationRequest` field.
_CAPABILITY_FLAGS = {
    "variables": ("variableSubset", "variable subsetting"),
    "bounding_box": ("bboxSubset", "bounding box subsetting"),
    "temporal": ("temporalSubset", "temporal subsetting"),
    "crs": ("reproject", "reprojection"),
    "concatenate": ("concatenate", "concatenation"),
}


def _import_harmony() -> Any:
    try:
        import harmony  # noqa: PLC0415
    except ImportError as err:
        msg = (
            "The Harmony transformation backend requires harmony-py. "
            "Install it with `pip install earthaccess[harmony]`."
        )
        raise ImportError(msg) from err
    return harmony


class HarmonyBackend(TransformationBackend):
    """Run transformations with [NASA Harmony](https://harmony.earthdata.nasa.gov).

    Backend options (passed via `backend_options` to `earthaccess.transform`):

    * **link_type**: (str) Type of output links to request: `"https"` (default),
      `"http"`, or `"s3"`.
    * **check_interval**: (float) Seconds between job status checks. Default: 3.
    * **env**: (`harmony.Environment`) Harmony environment to use. Defaults to the
      environment matching the earthaccess login system (PROD or UAT).

    Any `TransformationRequest.options` are passed straight through as keyword
    arguments to `harmony.Request`, e.g. `skip_preview=True`, `ignore_errors=True`,
    `granule_name=[...]`, `max_results=10`, `labels=[...]`.
    """

    name: ClassVar[str] = "harmony"

    def __init__(
        self,
        auth: Auth,
        *,
        link_type: str = "https",
        check_interval: float = 3.0,
        env: harmony.Environment | None = None,
    ) -> None:
        super().__init__(auth)
        self._harmony = _import_harmony()
        self._link_type = self._harmony.LinkType(link_type)

        if env is None:
            env = self._environment_for(auth)

        token = auth.token["access_token"] if auth.token else None
        self.client = self._harmony.Client(
            env=env,
            token=token,
            auth=None if token else (auth.username, auth.password),
            check_interval=check_interval,
            should_validate_auth=False,
        )

    def _environment_for(self, auth: Auth) -> harmony.Environment:
        environments = {
            PROD.edl_hostname: self._harmony.Environment.PROD,
            UAT.edl_hostname: self._harmony.Environment.UAT,
        }
        try:
            return environments[auth.system.edl_hostname]
        except KeyError:
            msg = (
                f"No Harmony environment is known for {auth.system.edl_hostname}; "
                "pass `env` in backend_options."
            )
            raise ValueError(msg) from None

    def capabilities(self, collection_id: str) -> dict[str, Any]:
        """Return Harmony's capabilities document for a collection.

        See <https://harmony.earthdata.nasa.gov/docs#get-collection-capabilities>.
        """
        request = self._harmony.CapabilitiesRequest(
            collection_id=collection_id,
            capabilities_version="2",
        )
        return self.client.submit(request)

    def validate(
        self,
        request: TransformationRequest,
        capabilities: dict[str, Any],
    ) -> None:
        """Check that Harmony can fulfill `request` for its collection.

        Only capabilities present in the capabilities document are checked, so
        that Harmony remains the final judge of anything not advertised there.
        """
        problems = []

        if not capabilities.get("services", True):
            problems.append("no Harmony services are configured for the collection")

        for attribute, (flag, description) in _CAPABILITY_FLAGS.items():
            value = getattr(request, attribute)
            if (
                value is not None
                and value is not False
                and capabilities.get(flag) is False
            ):
                problems.append(f"{description} is not supported")

        output_formats = capabilities.get("outputFormats")
        if (
            request.output_format
            and output_formats
            and request.output_format not in output_formats
        ):
            problems.append(
                f"output format {request.output_format!r} is not supported; "
                f"supported formats: {', '.join(output_formats)}"
            )

        known_variables = {
            v["name"].lstrip("/")
            for v in capabilities.get("variables") or []
            if v.get("name")
        }
        if request.variables and known_variables:
            unknown = [
                v
                for v in request.variables
                if v != _ALL_VARIABLES
                and not _VARIABLE_CONCEPT_ID.match(v)
                and v.lstrip("/") not in known_variables
            ]
            if unknown:
                problems.append(f"unknown variables: {', '.join(unknown)}")

        if problems:
            msg = (
                f"Harmony cannot transform collection {request.collection_id}: "
                + "; ".join(problems)
            )
            raise TransformationNotSupported(msg)

    def build_request(self, request: TransformationRequest) -> harmony.Request:
        """Translate a `TransformationRequest` into a `harmony.Request`."""
        kwargs: dict[str, Any] = {
            "granule_id": request.granule_ids,
            "format": request.output_format,
            "crs": request.crs,
            "concatenate": request.concatenate,
        }

        if request.variables:
            kwargs["variables"] = list(request.variables)
        if request.bounding_box is not None:
            kwargs["spatial"] = self._harmony.BBox(*request.bounding_box)
        if request.temporal is not None:
            start, stop = request.temporal
            kwargs["temporal"] = {
                key: value
                for key, value in (("start", start), ("stop", stop))
                if value is not None
            }

        options = dict(request.options)
        options["labels"] = [*options.get("labels", []), DEFAULT_LABEL]
        kwargs.update(options)

        harmony_request = self._harmony.Request(
            collection=self._harmony.Collection(id=request.collection_id),
            **{k: v for k, v in kwargs.items() if v is not None},
        )
        if not harmony_request.is_valid():
            msg = "Invalid Harmony request: " + "; ".join(
                harmony_request.error_messages()
            )
            raise ValueError(msg)
        return harmony_request

    def submit(self, request: TransformationRequest) -> str:
        """Submit `request` to Harmony and return the job ID."""
        response = self.client.submit(self.build_request(request))
        # Harmony returns the full job document instead of its ID when a job
        # completes synchronously.
        return response["jobID"] if isinstance(response, dict) else response

    def wait(self, job_id: str, *, show_progress: bool) -> tuple[str, str]:
        """Block until the Harmony job finishes and return `(status, message)`."""
        from harmony.client import ProcessingFailedException  # noqa: PLC0415

        try:
            self.client.wait_for_processing(job_id, show_progress=show_progress)
        except ProcessingFailedException as err:
            raise TransformationFailed(job_id, f"Harmony job failed: {err}") from err

        _, job_status, message = self.client.progress(job_id)

        if job_status not in SUCCESSFUL_STATUSES:
            hint = (
                " Large jobs are paused for preview; pass skip_preview=True to avoid this."
                if job_status == "paused"
                else ""
            )
            msg = f"Harmony job {job_id} ended with status {job_status!r}: {message}.{hint}"
            raise TransformationFailed(job_id, msg)

        if job_status == "complete_with_errors":
            logger.warning("Harmony job %s completed with errors: %s", job_id, message)

        return job_status, message

    def result_urls(self, job_id: str) -> list[str]:
        """Return the output URLs of a finished Harmony job."""
        return list(self.client.result_urls(job_id, link_type=self._link_type))

    def download(self, urls: list[str], local_path: Path, *, force: bool) -> list[Path]:
        """Download Harmony job outputs into `local_path`.

        Zarr stores cannot be downloaded as files and are skipped; open them from
        the result URLs instead.
        """
        futures = []
        for url in urls:
            if url.endswith("zarr"):
                logger.warning("Skipping download of Zarr store %s", url)
                continue
            futures.append(
                self.client.download(url, directory=str(local_path), overwrite=force)
            )
        return [Path(future.result()) for future in futures]
