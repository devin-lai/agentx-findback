"""Record whether a video is a camera capture that may still be growing.

Existing rows are imported files, so both columns stay null.
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("videos") as batch:
        batch.add_column(sa.Column("live_status", sa.String(20), nullable=True))
        batch.add_column(sa.Column("capture", sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table("videos") as batch:
        batch.drop_column("capture")
        batch.drop_column("live_status")
