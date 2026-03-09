from pathlib import Path

from strap.database import Database


def test_sanitize_table_name_prefixes_digit_started_names():
    assert Database._sanitize_table_name("60_common_solvents-TEA-LCA") == "t_60_common_solvents_tea_lca"


def test_get_database_caches_by_resolved_data_dir(tmp_path, monkeypatch):
    import strap.database as database_module

    dir_one = tmp_path / "data-one"
    dir_two = tmp_path / "data-two"
    dir_one.mkdir()
    dir_two.mkdir()
    (dir_one / "sample.csv").write_text("value\n1\n", encoding="utf-8")
    (dir_two / "sample.csv").write_text("value\n2\n", encoding="utf-8")

    monkeypatch.setattr(database_module, "_database", None)
    monkeypatch.setattr(database_module, "_database_cache", {})

    db_one = database_module.get_database(dir_one)
    db_one_again = database_module.get_database(Path(dir_one))
    db_two = database_module.get_database(dir_two)

    assert db_one is db_one_again
    assert db_one is not db_two
    assert db_one.conn.execute("SELECT value FROM sample").fetchone()[0] == 1
    assert db_two.conn.execute("SELECT value FROM sample").fetchone()[0] == 2
