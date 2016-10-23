from distutils.core import setup, Extension


SBCXX_module = Extension('_surface_Brightness_Profiles_CXX',
                           sources=['surface_Brightness_Profiles_CXX_wrap.cxx', 'surface_Brightness_Profiles_CXX.cxx'],
                           )

setup (name = 'example',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig example from docs""",
       ext_modules = [SBCXX_module],
       py_modules = ["surface_Brightness_Profiles_CXX"],
       )



