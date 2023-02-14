// RUN: disc-opt --disc-transform-dialect-interpreter -split-input-file %s | FileCheck %s

#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @main
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: tensor<?x?xf32>, %[[ARG3:.*]]: tensor<?x?xf32>)
func.func @main(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %k = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %n = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  // CHECK: %[[T0:.*]] = scf.for %[[IV:.*]] = %c0 to %[[K:.*]] step %c512 iter_args(%[[V0:.*]] = %[[ARG3]])
  %0:2 = scf.for %arg4 = %c0 to %k step %c512 iter_args(%arg5 = %arg0, %arg6 = %arg3) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %1 = affine.min #map(%arg4)[%k]
    %extracted_slice = tensor.extract_slice %arg0[0, %arg4] [%m, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, 0] [%1, %n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %extracted_slice_3 = tensor.extract_slice %arg5[0, 0] [%m, %n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    // CHECK: %[[R0:.*]] = tensor.extract_slice %[[V0]]
    // CHECK: linalg.matmul
    // CHECK-SAME: outs(%[[R0]] : tensor<?x?xf32>)
    %2 = linalg.matmul {disc.transform.name = "dot_general"} ins(%extracted_slice, %extracted_slice_2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_3 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %2 into %arg5[0, 0] [%m, %n] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    %extracted_slice_4 = tensor.extract_slice %arg6[0, 0] [%m, %n] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %3 = arith.addi %arg4, %c512 : index
    %4 = arith.cmpi sge, %3, %k : index
    %5 = disc_linalg_ext.conditional_generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%4, %2 : i1, tensor<?x?xf32>) outs(%extracted_slice_4 : tensor<?x?xf32>) attrs =  {disc.device = "cpu", disc.transform.name = "maximum"} {
    ^bb0(%in: i1, %in_6: f32, %out: f32):
      %6 = arith.addf %in_6, %in_6 : f32
      disc_linalg_ext.yield %6 : f32
    } -> tensor<?x?xf32>
    %inserted_slice_5 = tensor.insert_slice %5 into %arg6[0, 0] [%m, %n] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    scf.yield %inserted_slice, %inserted_slice_5 : tensor<?x?xf32>, tensor<?x?xf32>
  }
  return %0#1 : tensor<?x?xf32>
}

transform.structured.canonicalized_sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["scf.for"]} in %arg0
  transform.disc.replace_and_remove_loop_iter_arg %0 { from = 0 : i64, to = 1 : i64}
}