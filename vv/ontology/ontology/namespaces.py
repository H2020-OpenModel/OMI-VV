from rdflib import Namespace

MIO=Namespace("http://www.ddmd.io/mio#")
CUDS=Namespace("http://www.ddmd.io/mio/cuds#")
MISO=Namespace("http://www.ddmd.io/miso/")
# Export the CUDS namespace for direct import
__all__ = ["CUDS"]
