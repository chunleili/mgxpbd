// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2015 Alec Jacobson
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#include "piecewise_constant_winding_number.h"
#include "unique_edge_map.h"
#include "PI.h"

template <
  typename DerivedF,
  typename DeriveduE,
  typename DeriveduEC,
  typename DeriveduEE>
IGL_INLINE bool igl::piecewise_constant_winding_number(
  const Eigen::MatrixBase<DerivedF>& F,
  const Eigen::MatrixBase<DeriveduE>& uE,
  const Eigen::MatrixBase<DeriveduEC>& uEC,
  const Eigen::MatrixBase<DeriveduEE>& uEE)
{
  const size_t num_faces = F.rows();
  const size_t num_edges = uE.rows();
  const auto e2f = [&](size_t ei) { return ei % num_faces; };
  const auto is_consistent = [&](size_t fid, size_t s, size_t d)
  {
    if ((size_t)F(fid, 0) == s && (size_t)F(fid, 1) == d) return true;
    if ((size_t)F(fid, 1) == s && (size_t)F(fid, 2) == d) return true;
    if ((size_t)F(fid, 2) == s && (size_t)F(fid, 0) == d) return true;

    if ((size_t)F(fid, 0) == d && (size_t)F(fid, 1) == s) return false;
    if ((size_t)F(fid, 1) == d && (size_t)F(fid, 2) == s) return false;
    if ((size_t)F(fid, 2) == d && (size_t)F(fid, 0) == s) return false;
    throw "Invalid face!!";
  };
  for (size_t i=0; i<num_edges; i++)
  {
    const size_t s = uE(i,0);
    const size_t d = uE(i,1);
    int count=0;
    //for (const auto& ei : uE2E[i])
    for(size_t j = uEC(i);j<uEC(i+1);j++)
    {
      const size_t ei = uEE(j);
      const size_t fid = e2f(ei);
      if (is_consistent(fid, s, d))
      {
        count++;
      }
      else
      {
        count--;
      }
    }
    if (count != 0)
    {
      return false;
    }
  }
  return true;
}

template <typename DerivedF>
IGL_INLINE bool igl::piecewise_constant_winding_number(
  const Eigen::MatrixBase<DerivedF>& F)
{
  Eigen::Matrix<typename DerivedF::Scalar,Eigen::Dynamic,2> E, uE;
  Eigen::Matrix<typename DerivedF::Scalar,Eigen::Dynamic,1> EMAP, uEC, uEE;
  unique_edge_map(F, E, uE, EMAP, uEC, uEE);
  return piecewise_constant_winding_number(F,uE,uEC,uEE);
}

#ifdef IGL_STATIC_LIBRARY
// Explicit template instantiation
// generated by autoexplicit.sh
template bool igl::piecewise_constant_winding_number<Eigen::Matrix<int, -1, 3, 0, -1, 3> >(Eigen::MatrixBase<Eigen::Matrix<int, -1, 3, 0, -1, 3> > const&);
// generated by autoexplicit.sh
template bool igl::piecewise_constant_winding_number<Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<int, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, 1, 0, -1, 1> > const&);
#ifdef WIN32
#endif
#endif
