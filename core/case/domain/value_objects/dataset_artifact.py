from core.case.domain.enums import DataFormat, DataCompression
from typing import Optional
from core.case.domain.entities.base_artifact import BaseArtifactPointer


class DataArtifactPointer (BaseArtifactPointer) :
    data_format : DataFormat
    compression : Optional[DataCompression] = None
    record_count : Optional[int] = None
    column_count : Optional [int] = None
