from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        connect_args = (
            {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
        )
        self.engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def configure_sqlite(connection, _record):
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()

        self.session = sessionmaker(self.engine, expire_on_commit=False)

    def migrate(self) -> None:
        cfg = Config()
        cfg.set_main_option("script_location", str(Path(__file__).parents[1] / "migrations"))
        cfg.set_main_option("sqlalchemy.url", self.url.replace("%", "%%"))
        with self.engine.begin() as connection:
            cfg.attributes["connection"] = connection
            command.upgrade(cfg, "head")
