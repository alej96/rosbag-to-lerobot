from bag2lerobot.extract import inspect_bag
from tests.conftest import BagSpec


def test_inspect_bag_burst_topic_rate_is_sane(make_bag):
    """A topic republished as a burst (nearly all messages sharing one bag timestamp,
    like a static /tf replay) must not report an absurd rate: dividing the message count
    by the tiny first/last span used to blow up to the tens-of-millions of Hz, instead of
    the median gap collapsing it to the usual "no reliable rate" 0.0."""
    bag = make_bag(spec=BagSpec(burst_topic="/tf", burst_count=50))

    topics = {t.topic: t for t in inspect_bag(bag)}

    assert topics["/tf"].count == 50
    assert topics["/tf"].rate_hz < 1000.0


def test_inspect_bag_steady_topic_rate_is_accurate(make_bag):
    bag = make_bag(spec=BagSpec(follower_hz=100.0))

    topics = {t.topic: t for t in inspect_bag(bag)}

    assert 90.0 < topics["/follower/joint_states"].rate_hz < 110.0
