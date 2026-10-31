from convert_html import html_to_md, sanity_check


def test_html_to_md_converts_apple_notes_html():
    html = (
        "<div><h1>Trip</h1></div><div><br></div><div></div>"
        "<div><b>Pack</b> list:</div><ul><li>passport</li><li>adapter</li></ul>"
        '<div><a href="https://example.com">site</a></div>'
    )
    md = html_to_md(html)
    assert md.startswith("# Trip")
    assert "**Pack** list:" in md
    assert "- passport" in md and "- adapter" in md
    assert "[site](https://example.com)" in md
    assert "\n\n\n" not in md


def test_sanity_check_flags():
    assert sanity_check("n", "t", "", '<img src="data:image/png;base64,AAA">') == ["IMAGE_ONLY"]
    assert sanity_check("n", "t", "", "<div></div>") == ["EMPTY"]
    assert sanity_check("n", "t", "x" * 100_001, "")[0].startswith("OVERSIZED")
    assert sanity_check("n", "t", "é" * 60, "")[0].startswith("HIGH_NON_ASCII")
    assert sanity_check("n", "t", "plain note text", "") == []
