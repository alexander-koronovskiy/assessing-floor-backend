import pytest

from main import create_app


@pytest.fixture(autouse=True, scope='class')
def service_class():
    return create_app()


def pytest_addoption(parser):
    parser.addoption(
        '--correct-plan-path',
        action='store',
        default='tests/assets/correct_plan.jpg',
        help='path to correct plan image'
    )
    parser.addoption(
        '--valid-resp-path',
        action='store',
        default='tests/assets/response_without_pretrain.json',
        help='path to valid response data'
    )
    parser.addoption(
        '--not-a-plan-path',
        action='store',
        default='tests/assets/not_a_plan.jpg',
        help='path to valid response data'
    )


def pytest_configure(config):
    pytest.correct_plan_path = config.getoption('--correct-plan-path')
    pytest.valid_response_path = config.getoption('--valid-resp-path')
    pytest.sample_image_path = config.getoption('--not-a-plan-path')
