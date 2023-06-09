import smaller3d.models.resunet as resunet
import smaller3d.models.res16unet as res16unet
from smaller3d.models.res16unet import Res16UNet34C, Res16UNet34A, Res16UNet14A,Res16UNet34C_HALF,Res16UNet34C_HALF_HALF
from smaller3d.models.conditional_random_fields import BilateralCRF, TrilateralCRF

MODELS = []


def add_models(module):
    MODELS.extend([getattr(module, a) for a in dir(module) if "Net" in a])


add_models(resunet)
add_models(res16unet)

WRAPPERS = [BilateralCRF, TrilateralCRF]


def get_models():
    """Returns a tuple of sample models."""
    return MODELS


def get_wrappers():
    return WRAPPERS


def load_model(name):
    """Creates and returns an instance of the model given its class name.
  """
    # Find the model class from its name
    all_models = get_models()
    mdict = {model.__name__: model for model in all_models}
    if name not in mdict:
        print("Invalid model index. Options are:")
        # Display a list of valid model names
        for model in all_models:
            print(f"\t* {model.__name__}")
        return None
    NetClass = mdict[name]

    return NetClass


def load_wrapper(name):
    """Creates and returns an instance of the model given its class name.
  """
    # Find the model class from its name
    all_wrappers = get_wrappers()
    mdict = {wrapper.__name__: wrapper for wrapper in all_wrappers}
    if name not in mdict:
        print("Invalid wrapper index. Options are:")
        # Display a list of valid model names
        for wrapper in all_wrappers:
            print(f"\t* {wrapper.__name__}")
        return None
    WrapperClass = mdict[name]

    return WrapperClass
