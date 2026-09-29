"""Initial video memory schema

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "videos",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("original_name", sa.String(length=255), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=False),
        sa.Column("source_path", sa.String(length=255), nullable=False),
        sa.Column("media_path", sa.String(length=255), nullable=False),
        sa.Column("regions", sa.JSON(), nullable=False),
        sa.Column("is_fixture", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "index_runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("video_id", sa.String(length=32), nullable=False),
        sa.Column("backend", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("object_ids", sa.JSON(), nullable=False),
        sa.Column("regions", sa.JSON(), nullable=False),
        sa.Column("processed_ms", sa.Integer(), nullable=False),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.Column("elapsed_seconds", sa.Float(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("index_runs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_index_runs_status"), ["status"], unique=False)
        batch_op.create_index(batch_op.f("ix_index_runs_video_id"), ["video_id"], unique=False)

    op.create_table(
        "objects",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("video_id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("label", sa.String(length=80), nullable=False),
        sa.Column("registered_at_ms", sa.Integer(), nullable=False),
        sa.Column("box", sa.JSON(), nullable=False),
        sa.Column("reference_path", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("video_id", "name", name="uq_object_video_name"),
    )
    with op.batch_alter_table("objects", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_objects_video_id"), ["video_id"], unique=False)

    op.create_table(
        "observations",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("object_id", sa.String(length=32), nullable=False),
        sa.Column("at_ms", sa.Integer(), nullable=False),
        sa.Column("box", sa.JSON(), nullable=True),
        sa.Column("zone", sa.String(length=40), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("reason", sa.String(length=60), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column("visible", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["object_id"], ["objects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["index_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "object_id", "at_ms", name="uq_observation_time"),
    )
    with op.batch_alter_table("observations", schema=None) as batch_op:
        batch_op.create_index(
            "ix_observation_history", ["run_id", "object_id", "at_ms"], unique=False
        )

    op.create_table(
        "queries",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("cutoff_ms", sa.Integer(), nullable=False),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["index_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("queries", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_queries_run_id"), ["run_id"], unique=False)

    op.create_table(
        "events",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("object_id", sa.String(length=32), nullable=False),
        sa.Column("at_ms", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("zone", sa.String(length=40), nullable=True),
        sa.Column("previous_zone", sa.String(length=40), nullable=True),
        sa.Column("reason", sa.String(length=60), nullable=True),
        sa.Column("observation_id", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["object_id"], ["objects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["observation_id"], ["observations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["index_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("events", schema=None) as batch_op:
        batch_op.create_index("ix_event_history", ["run_id", "object_id", "at_ms"], unique=False)


def downgrade():
    with op.batch_alter_table("events", schema=None) as batch_op:
        batch_op.drop_index("ix_event_history")

    op.drop_table("events")
    with op.batch_alter_table("queries", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_queries_run_id"))

    op.drop_table("queries")
    with op.batch_alter_table("observations", schema=None) as batch_op:
        batch_op.drop_index("ix_observation_history")

    op.drop_table("observations")
    with op.batch_alter_table("objects", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_objects_video_id"))

    op.drop_table("objects")
    with op.batch_alter_table("index_runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_index_runs_video_id"))
        batch_op.drop_index(batch_op.f("ix_index_runs_status"))

    op.drop_table("index_runs")
    op.drop_table("videos")
