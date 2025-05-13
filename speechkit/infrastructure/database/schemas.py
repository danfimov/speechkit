import datetime as dt
import uuid

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, as_declarative, mapped_column, relationship

from speechkit.domain.dto import task_status


sa_metadata = sa.MetaData(
    naming_convention={
        'ix': 'ix_%(column_0_label)s',
        'uq': 'uq_%(table_name)s_%(column_0_name)s',
        'ck': 'ck_%(table_name)s_%(constraint_name)s',
        'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s',
        'pk': 'pk_%(table_name)s',
    },
)


@as_declarative(metadata=sa_metadata)
class BaseTableSchema:
    created_at: Mapped[dt.datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=dt.datetime.now,
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=dt.datetime.now,
        onupdate=dt.datetime.now,
    )


class ServiceToken(BaseTableSchema):
    __tablename__ = 'service_token'

    id: Mapped[uuid.UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    hashed_token: Mapped[str] = mapped_column(sa.String(255), nullable=False)


class Task(BaseTableSchema):
    __tablename__ = 'task'

    id: Mapped[uuid.UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[task_status.TaskStatus] = mapped_column(sa.Integer, nullable=False)
    result: Mapped[str] = mapped_column(postgresql.TEXT, default='')

    files: Mapped[list['FileMetadata']] = relationship(back_populates='task')


class FileMetadata(BaseTableSchema):
    __tablename__ = 'file_metadata'

    id: Mapped[uuid.UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(sa.String(255), nullable=True)
    content_type: Mapped[str] = mapped_column(sa.String(255), nullable=True)
    task_id: Mapped[uuid.UUID] = mapped_column(
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey('task.id'),
        nullable=False,
        index=True,
    )

    task: Mapped['Task'] = relationship(back_populates='files')
