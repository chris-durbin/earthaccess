"""Backend-agnostic building blocks for transformation plugins."""

from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from earthaccess.auth import Auth

logger = logging.getLogger(__name__)


@dataclass
class TransformationRequest:
    """A backend-agnostic description of a transformation for a single collection.

    Attributes:
        collection_id: CMR concept ID of the collection to transform.
        granule_ids: CMR concept IDs of the granules to transform. When `None`,
            the backend selects granules from the whole collection (constrained by
            `bounding_box` and `temporal`).
        variables: Names (or CMR variable concept IDs) of the variables to subset.
        bounding_box: Spatial subset as `(west, south, east, north)` in degrees.
        temporal: Temporal subset as `(start, stop)`; either end may be `None`.
        output_format: Output MIME type, e.g. `"application/x-netcdf4"`.
        crs: Output coordinate reference system, e.g. `"EPSG:4326"`.
        concatenate: Whether outputs should be concatenated into a single file.
        options: Additional backend-specific options, passed through unchanged.
    """

    collection_id: str
    granule_ids: list[str] | None = None
    variables: list[str] | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    temporal: tuple[datetime.datetime | None, datetime.datetime | None] | None = None
    output_format: str | None = None
    crs: str | None = None
    concatenate: bool | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformationResult:
    """The outcome of a transformation job.

    Attributes:
        backend: Name of the backend that ran the job.
        collection_id: CMR concept ID of the transformed collection.
        job_id: Backend-specific identifier of the job.
        status: Final job status as reported by the backend.
        message: Final job message as reported by the backend.
        urls: URLs of the job's output files.
        files: Local paths of the downloaded output files (empty if not downloaded).
    """

    backend: str
    collection_id: str
    job_id: str
    status: str
    message: str = ""
    urls: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)


class TransformationBackend(ABC):
    """Base class for transformation backends.

    A backend turns a `TransformationRequest` into a job on some transformation
    service. The overall workflow is implemented by `run`, which calls the abstract
    steps below in order: `capabilities`, `validate`, `submit`, `wait`,
    `result_urls` and, optionally, `download`.

    Third-party backends can be made available to `earthaccess.transform` either by
    calling `earthaccess.transformations.register_backend`, or by declaring an entry
    point in the `earthaccess.transformations` group, e.g. in `pyproject.toml`:

    ```toml
    [project.entry-points."earthaccess.transformations"]
    mybackend = "my_package.module:MyBackend"
    ```
    """

    name: ClassVar[str]

    def __init__(self, auth: Auth, **options: Any) -> None:  # noqa: ARG002
        self.auth = auth

    @abstractmethod
    def capabilities(self, collection_id: str) -> dict[str, Any]:
        """Return the transformation capabilities of a collection."""

    def validate(  # noqa: B027 - optional hook; backends need not validate
        self,
        request: TransformationRequest,
        capabilities: dict[str, Any],
    ) -> None:
        """Check `request` against `capabilities` before it is submitted.

        Raises:
            TransformationNotSupported: The collection cannot fulfill the request.
        """

    @abstractmethod
    def submit(self, request: TransformationRequest) -> str:
        """Submit a transformation request and return the job ID."""

    @abstractmethod
    def wait(self, job_id: str, *, show_progress: bool) -> tuple[str, str]:
        """Block until a job finishes and return its final `(status, message)`.

        Raises:
            TransformationFailed: The job failed or did not run to completion.
        """

    @abstractmethod
    def result_urls(self, job_id: str) -> list[str]:
        """Return the URLs of a finished job's output files."""

    @abstractmethod
    def download(self, urls: list[str], local_path: Path, *, force: bool) -> list[Path]:
        """Download a finished job's output files into `local_path`."""

    def run(
        self,
        request: TransformationRequest,
        *,
        local_path: Path | None,
        show_progress: bool,
        validate: bool = True,
        force: bool = False,
    ) -> TransformationResult:
        """Run a transformation end to end.

        Parameters:
            request: The transformation to run.
            local_path: Directory to download outputs into, or `None` to skip
                downloading and only return output URLs.
            show_progress: Whether to display job progress.
            validate: Whether to check the request against the collection's
                capabilities before submitting it.
            force: Re-download outputs that already exist locally.

        Returns:
            The result of the transformation job.
        """
        if validate:
            capabilities = self.capabilities(request.collection_id)
            self.validate(request, capabilities)

        job_id = self.submit(request)
        logger.info("Submitted %s job %s", self.name, job_id)

        status, message = self.wait(job_id, show_progress=show_progress)
        logger.info("%s job %s finished with status %s", self.name, job_id, status)

        result = TransformationResult(
            backend=self.name,
            collection_id=request.collection_id,
            job_id=job_id,
            status=status,
            message=message,
            urls=self.result_urls(job_id),
        )

        if local_path is not None and result.urls:
            local_path.mkdir(parents=True, exist_ok=True)
            result.files = self.download(result.urls, local_path, force=force)

        return result


def normalize_temporal(
    temporal: Sequence[str | datetime.datetime | None],
) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """Convert a `(start, stop)` pair of ISO 8601 strings/datetimes into datetimes.

    Timezone-aware values are converted to UTC; naive values are assumed to be UTC.
    As with `earthaccess.search_data`, a date-only `stop` string includes that
    entire day (i.e. the time is set to `23:59:59`).
    """
    if len(temporal) != 2:  # noqa: PLR2004
        msg = f"temporal must be a (start, stop) pair, got {temporal!r}"
        raise ValueError(msg)

    def to_datetime(
        value: str | datetime.datetime | None,
        *,
        end_of_day: bool,
    ) -> datetime.datetime | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime.datetime):
            dt = value
        else:
            dt = datetime.datetime.fromisoformat(value)
            if end_of_day and len(value) == len("YYYY-MM-DD"):
                dt = dt.replace(hour=23, minute=59, second=59)
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.UTC).replace(tzinfo=None)
        return dt

    start = to_datetime(temporal[0], end_of_day=False)
    stop = to_datetime(temporal[1], end_of_day=True)

    if start is not None and stop is not None and start > stop:
        msg = f"temporal start {start} is after stop {stop}"
        raise ValueError(msg)

    return start, stop
