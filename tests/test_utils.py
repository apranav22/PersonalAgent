from utils import load_manifest


def test_load_manifest_parses_ids_titles_and_timestamps(tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "p1|||Grocery list|||1700000000.75\n"
        "\n"
        "p2|||Ideas ||| \n"
        "p3|||Bad ts|||not-a-number\n"
        "malformed line without separator\n",
        encoding="utf-8",
    )
    m = load_manifest(manifest)
    assert m == {
        "p1": {"title": "Grocery list", "modified_at": 1700000000},
        "p2": {"title": "Ideas", "modified_at": None},
        "p3": {"title": "Bad ts", "modified_at": None},
    }


def test_load_manifest_two_field_format(tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("note_1|||Journal\n", encoding="utf-8")
    assert load_manifest(manifest) == {"note_1": {"title": "Journal", "modified_at": None}}
