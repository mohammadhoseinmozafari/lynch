from case.domain.enums import DataFormat, DataCompression
from typing import Optional
from case.domain.value_objects.base_artifact import BaseArtifactPointer


class DataArtifactPointer (BaseArtifactPointer) :
    data_format : DataFormat
    compression : Optional[DataCompression] = None
    record_count : Optional[int] = None
    column_count : Optional [int] = None
