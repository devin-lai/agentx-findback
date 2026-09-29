"""Keep the camera transform that a compensated region was read through.

Existing rows were observed at the registration pose, where the transform is the identity and
carries no information, so they stay null.
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("observations") as batch:
        batch.add_column(sa.Column("scene_transform", sa.JSON(), nullable=True))


def downgrade():
    with op.batch_alter_table("observations") as batch:
        batch.drop_column("scene_transform")
