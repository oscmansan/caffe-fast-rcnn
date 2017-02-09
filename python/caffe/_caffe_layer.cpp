#include <Python.h>  // NOLINT(build/include_alpha)

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/enum.hpp>
#include <numpy/arrayobject.h>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/python_layer.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

namespace bp = boost::python;

namespace caffe {

// For Python, for now, we'll just always use float as the type.
typedef float Dtype;
typedef float Mtype;
const int NPY_DTYPE = NPY_FLOAT32;

// Selecting mode.
void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }

void CheckContiguousArray(PyArrayObject* arr, string name,
    int channels, int height, int width) {
  if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
    throw std::runtime_error(name + " must be C contiguous");
  }
  if (PyArray_NDIM(arr) != 4) {
    throw std::runtime_error(name + " must be 4-d");
  }
  if (PyArray_TYPE(arr) != NPY_FLOAT32) {
    throw std::runtime_error(name + " must be float32");
  }
  if (PyArray_DIMS(arr)[1] != channels) {
    throw std::runtime_error(name + " has wrong number of channels");
  }
  if (PyArray_DIMS(arr)[2] != height) {
    throw std::runtime_error(name + " has wrong height");
  }
  if (PyArray_DIMS(arr)[3] != width) {
    throw std::runtime_error(name + " has wrong width");
  }
}

struct NdarrayConverterGenerator {
  template <typename T> struct apply;
};

template <>
struct NdarrayConverterGenerator::apply<Dtype*> {
  struct type {
    PyObject* operator() (Dtype* data) const {
      // Just store the data pointer, and add the shape information in postcall.
      return PyArray_SimpleNewFromData(0, NULL, NPY_DTYPE, data);
    }
    const PyTypeObject* get_pytype() {
      return &PyArray_Type;
    }
  };
};

struct NdarrayCallPolicies : public bp::default_call_policies {
  typedef NdarrayConverterGenerator result_converter;
  PyObject* postcall(PyObject* pyargs, PyObject* result) {
    bp::object pyblob = bp::extract<bp::tuple>(pyargs)()[0];
    shared_ptr<Blob<Dtype, Mtype> > blob =
      bp::extract<shared_ptr<Blob<Dtype, Mtype> > >(pyblob);
    // Free the temporary pointer-holding array, and construct a new one with
    // the shape information from the blob.
    void* data = PyArray_DATA(reinterpret_cast<PyArrayObject*>(result));
    Py_DECREF(result);
    const int num_axes = blob->num_axes();
    vector<npy_intp> dims(blob->shape().begin(), blob->shape().end());
    PyObject *arr_obj = PyArray_SimpleNewFromData(num_axes, dims.data(),
                                                  NPY_FLOAT32, data);
    // SetBaseObject steals a ref, so we need to INCREF.
    Py_INCREF(pyblob.ptr());
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(arr_obj),
        pyblob.ptr());
    return arr_obj;
  }
};

bp::object Blob_Reshape(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("Blob.reshape takes no kwargs");
  }
  Blob<Dtype, Mtype>* self = bp::extract<Blob<Dtype, Mtype>*>(args[0]);
  vector<int> shape(bp::len(args) - 1);
  for (int i = 1; i < bp::len(args); ++i) {
    shape[i - 1] = bp::extract<int>(args[i]);
  }
  self->Reshape(shape);
  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}

bp::object BlobVec_add_blob(bp::tuple args, bp::dict kwargs) {
  if (bp::len(kwargs) > 0) {
    throw std::runtime_error("BlobVec.add_blob takes no kwargs");
  }
  typedef vector<shared_ptr<Blob<Dtype,Mtype> > > BlobVec;
  BlobVec* self = bp::extract<BlobVec*>(args[0]);
  vector<int> shape(bp::len(args) - 1);
  for (int i = 1; i < bp::len(args); ++i) {
    shape[i - 1] = bp::extract<int>(args[i]);
  }
  self->push_back(shared_ptr<Blob<Dtype,Mtype> >(new Blob<Dtype,Mtype>(shape)));
  // We need to explicitly return None to use bp::raw_function.
  return bp::object();
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolveOverloads, Solve, 0, 1);

BOOST_PYTHON_MODULE(_caffe) {
  // below, we prepend an underscore to methods that will be replaced
  // in Python

  bp::scope().attr("__version__") = AS_STRING(CAFFE_VERSION);

  // Caffe utility functions
  bp::def("set_mode_cpu", &set_mode_cpu);
  bp::def("set_mode_gpu", &set_mode_gpu);

  bp::def("layer_type_list", &LayerRegistry<Dtype, Mtype>::LayerTypeList);

  bp::enum_<Phase>("Phase")
    .value("TRAIN", caffe::TRAIN)
    .value("TEST", caffe::TEST)
    .export_values();

  bp::class_<Blob<Dtype, Mtype>, shared_ptr<Blob<Dtype, Mtype> >, boost::noncopyable>(
    "Blob", bp::no_init)
    .add_property("shape",
        bp::make_function(
            static_cast<const vector<int>& (Blob<Dtype, Mtype>::*)() const>(
                &Blob<Dtype, Mtype>::shape),
            bp::return_value_policy<bp::copy_const_reference>()))
    .add_property("num",      &Blob<Dtype, Mtype>::num)
    .add_property("channels", &Blob<Dtype, Mtype>::channels)
    .add_property("height",   &Blob<Dtype, Mtype>::height)
    .add_property("width",    &Blob<Dtype, Mtype>::width)
    .add_property("count",    static_cast<int (Blob<Dtype, Mtype>::*)() const>(
        &Blob<Dtype, Mtype>::count))
    .def("reshape",           bp::raw_function(&Blob_Reshape))
    .add_property("data",     bp::make_function(&Blob<Dtype, Mtype>::mutable_cpu_data,
          NdarrayCallPolicies()))
    .add_property("diff",     bp::make_function(&Blob<Dtype, Mtype>::mutable_cpu_diff,
          NdarrayCallPolicies()));

  bp::class_<Layer<Dtype, Mtype>, shared_ptr<PythonLayer<Dtype, Mtype> >,
    boost::noncopyable>("Layer", bp::init<const LayerParameter&>())
    .add_property("blobs", bp::make_function(&Layer<Dtype, Mtype>::blobs,
          bp::return_internal_reference<>()))
    .def("setup", &Layer<Dtype, Mtype>::LayerSetUp)
    .def("reshape", &Layer<Dtype, Mtype>::Reshape)
    .add_property("phase", bp::make_function(&Layer<Dtype, Mtype>::phase))
    .add_property("type", bp::make_function(&Layer<Dtype, Mtype>::type));
  bp::register_ptr_to_python<shared_ptr<Layer<Dtype, Mtype> > >();

  bp::class_<LayerParameter>("LayerParameter", bp::no_init);

  // vector wrappers for all the vector types we use
  bp::class_<vector<shared_ptr<Blob<Dtype, Mtype> > > >("BlobVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Blob<Dtype, Mtype> > >, true>())
    .def("add_blob", bp::raw_function(&BlobVec_add_blob));
  bp::class_<vector<Blob<Dtype, Mtype>*> >("RawBlobVec")
    .def(bp::vector_indexing_suite<vector<Blob<Dtype, Mtype>*>, true>());
  bp::class_<vector<shared_ptr<Layer<Dtype, Mtype> > > >("LayerVec")
    .def(bp::vector_indexing_suite<vector<shared_ptr<Layer<Dtype, Mtype> > >, true>());
  bp::class_<vector<string> >("StringVec")
    .def(bp::vector_indexing_suite<vector<string> >());
  bp::class_<vector<int> >("IntVec")
    .def(bp::vector_indexing_suite<vector<int> >());
  bp::class_<vector<Dtype> >("DtypeVec")
    .def(bp::vector_indexing_suite<vector<Dtype> >());
  bp::class_<vector<bool> >("BoolVec")
    .def(bp::vector_indexing_suite<vector<bool> >());

  // boost python expects a void (missing) return value, while import_array
  // returns NULL for python3. import_array1() forces a void return value.
  import_array1();
}

}  // namespace caffe
