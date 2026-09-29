"""Persist validated, non-authoritative visual reviews and their provenance."""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "reviews",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(32),
            sa.ForeignKey("index_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_reviews_run_id", "reviews", ["run_id"])


def downgrade():
    op.drop_index("ix_reviews_run_id", table_name="reviews")
    op.drop_table("reviews")
