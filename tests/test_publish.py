"""publish.py must never make anything public by accident."""

import pytest

import publish


def test_repos_are_private_unless_public_is_asked_for():
    args = publish.build_parser().parse_args(["--base", "b", "--chat", "c"])
    assert args.public is False


def test_a_private_deploy_must_bundle_its_weights():
    # the Hub cannot preload a Space's weights from a private model repo
    with pytest.raises(SystemExit, match="bundle-weights"):
        publish.main(["--base", "b", "--chat", "c"])


def test_space_requirements_leave_gradio_to_the_sdk_version():
    requirements, gradio = publish.app_requirements()
    assert gradio and "gradio" not in requirements.lower()
    assert "torch==" in requirements
