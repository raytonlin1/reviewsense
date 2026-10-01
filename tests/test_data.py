from reviewsense.data import Review, clean, load_sample, split_sentences


def test_sample_loads():
    reviews = load_sample()
    assert len(reviews) == 12 and len({r.business for r in reviews}) == 4
    assert all(isinstance(r, Review) and r.text and r.stars in {1, 2, 3, 4, 5} for r in reviews)


def test_meta_default_not_shared():
    a, b = Review("1", "x", "t"), Review("2", "y", "u")
    a.meta["k"] = 1
    assert b.meta == {}


def test_clean_html_entities_and_mojibake():
    assert clean("<p>Caf\u00c3\u00a9 was <b>great</b> &amp; cheap</p>\n\n  ok") == "Café was great & cheap ok"
    assert clean("great<br/>food") == "great food"                        # tags become spaces, words don't merge
    assert clean("I said \u00e2\u20ac\u0153wow\u00e2\u20ac\u009d") == 'I said "wow"'  # mojibake repaired, quotes straightened


def test_clean_leaves_plain_text_alone():
    assert clean("5 < 6 and the pho was good") == "5 < 6 and the pho was good"


def test_sentences_hard_cases():
    assert split_sentences("Dr. Li paid $4.50. It was great! Yes.") == ["Dr. Li paid $4.50.", "It was great!", "Yes."]
    assert split_sentences('He said "Great food." Then left.') == ['He said "Great food."', "Then left."]
    assert split_sentences("Wow!!! Would go again... Yes.") == ["Wow!!!", "Would go again...", "Yes."]
    assert split_sentences("i loved it. the soup was great.") == ["i loved it.", "the soup was great."]


def test_sentences_known_limits():
    # documented failures: no space after the period, and an emoji used as a full stop
    assert len(split_sentences("Great food.Terrible service.")) == 1
    assert len(split_sentences("Best tacos ever 🌮 Will be back!")) == 1
