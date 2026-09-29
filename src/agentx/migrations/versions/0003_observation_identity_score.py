"""Record the appearance-identity similarity of each learned observation."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("observations") as batch:
        batch.add_column(sa.Column("identity_score", sa.Float(), nullable=True))


def downgrade():
    with op.batch_alter_table("observations") as batch:
        batch.drop_column("identity_score")
