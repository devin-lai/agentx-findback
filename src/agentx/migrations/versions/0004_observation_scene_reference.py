"""Record how each observed frame related to the registration camera pose.

Existing rows were produced by the latching guard, which only ever observed a frame while the
camera was at its registered pose. Backfilling them as `registered` preserves their meaning.
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("observations") as batch:
        batch.add_column(
            sa.Column(
                "scene_reference",
                sa.String(length=20),
                nullable=False,
                server_default="registered",
            )
        )


def downgrade():
    with op.batch_alter_table("observations") as batch:
        batch.drop_column("scene_reference")
