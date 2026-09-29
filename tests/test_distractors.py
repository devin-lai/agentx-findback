from agentx.domain.contracts import Box, Detection
from agentx.vision.distractors import CollisionGuard, coverage


def obj(name, box, label="cup"):
    return dict(id=name, box=box.model_dump(), label=label)


def test_merge_with_companion_quarantines_identity_and_never_exports_companion():
    target_box = Box(x1=0.1, y1=0.1, x2=0.3, y2=0.4)
    other_box = Box(x1=0.5, y1=0.1, x2=0.7, y2=0.4)
    guard = CollisionGuard([obj("target", target_box)])
    guard.register_auxiliary([obj("companion", other_box)])
    separate = [Detection("target", target_box, 0.99), Detection("companion", other_box, 0.99)]
    assert guard.apply(separate) == [separate[0]]
    merged = Box(x1=0.1, y1=0.1, x2=0.7, y2=0.4)
    assert coverage(merged, other_box) == 1
    result = guard.apply([Detection("target", merged, 0.999), separate[1]])
    assert len(result) == 1 and result[0].box is None
    assert result[0].reason == "identity_ambiguous"
    # High scores and later non-overlap cannot silently relabel a lost identity.
    assert guard.apply(separate)[0].reason == "identity_ambiguous"


def test_different_categories_are_not_geometry_competitors():
    box = Box(x1=0.1, y1=0.1, x2=0.3, y2=0.4)
    guard = CollisionGuard([obj("cup", box), obj("book", box, "book")])
    results = [Detection("cup", box, 0.99), Detection("book", box, 0.99)]
    assert guard.apply(results) == results


def test_later_user_registration_supersedes_internal_track():
    a = Box(x1=0.1, y1=0.1, x2=0.3, y2=0.4)
    b = Box(x1=0.5, y1=0.1, x2=0.7, y2=0.4)
    public = [obj("a", a), obj("b", b)]
    guard = CollisionGuard(public)
    guard.register_auxiliary([obj("private-b", b)])
    results = [Detection("a", a, 0.99), Detection("b", b, 0.99), Detection("private-b", b, 0.99)]
    guard.retire_duplicates([public[1]], results)
    assert guard.retired == {"private-b"}
    assert guard.apply(results) == results[:2]


def test_same_category_registered_objects_can_also_lose_identity():
    box = Box(x1=0.1, y1=0.1, x2=0.3, y2=0.4)
    guard = CollisionGuard([obj("a", box), obj("b", box)])
    results = guard.apply([Detection("a", box, 0.99), Detection("b", box, 0.99)])
    assert all(d.box is None and d.reason == "identity_ambiguous" for d in results)


def test_existing_mask_ambiguity_is_preserved_without_a_positive_box():
    box = Box(x1=0.1, y1=0.1, x2=0.3, y2=0.4)
    guard = CollisionGuard([obj("target", box)])
    guard.apply([Detection("target", None, 0.99, "identity_ambiguous")])
    assert guard.apply([Detection("target", box, 0.999)])[0].box is None


def test_abrupt_mask_expansion_quarantines_identity_without_an_initial_companion():
    box = Box(x1=0.1, y1=0.1, x2=0.2, y2=0.3)
    merged = Box(x1=0.1, y1=0.1, x2=0.6, y2=0.5)
    guard = CollisionGuard([obj("target", box)])
    for _ in range(8):
        assert guard.apply([Detection("target", box, 0.999)])[0].box == box
    suspicious = guard.apply([Detection("target", merged, 0.999)])[0]
    assert suspicious.box is None and suspicious.reason == "identity_ambiguous"
    # A later small mask cannot re-establish which look-alike is the original.
    assert guard.apply([Detection("target", box, 0.999)])[0].reason == "identity_ambiguous"
    assert max(guard.accepted_areas["target"]) == (box.x2 - box.x1) * (box.y2 - box.y1)


def test_gradual_mask_size_change_keeps_confirmed_identity():
    first = Box(x1=0.1, y1=0.1, x2=0.2, y2=0.3)
    guard = CollisionGuard([obj("target", first)])
    for width in (0.13, 0.16, 0.19, 0.22):
        box = Box(x1=0.1, y1=0.1, x2=0.1 + width, y2=0.3)
        result = guard.apply([Detection("target", box, 0.999)])[0]
        assert result.box == box and result.reason is None


def test_registration_box_does_not_set_the_mask_size_baseline():
    small_prompt = Box(x1=0.1, y1=0.1, x2=0.12, y2=0.13)
    whole_object = Box(x1=0.1, y1=0.1, x2=0.3, y2=0.4)
    guard = CollisionGuard([obj("target", small_prompt)])
    for _ in range(3):
        result = guard.apply([Detection("target", whole_object, 0.999)])[0]
        assert result.box == whole_object and result.reason is None
