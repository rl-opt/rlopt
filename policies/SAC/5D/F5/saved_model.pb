?? 
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
?
7ActorDistributionNetwork/EncodingNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*H
shared_name97ActorDistributionNetwork/EncodingNetwork/dense_2/kernel
?
KActorDistributionNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOp7ActorDistributionNetwork/EncodingNetwork/dense_2/kernel*
_output_shapes
:	?*
dtype0
?
5ActorDistributionNetwork/EncodingNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense_2/bias
?
IActorDistributionNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense_2/bias*
_output_shapes	
:?*
dtype0
?
7ActorDistributionNetwork/EncodingNetwork/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*H
shared_name97ActorDistributionNetwork/EncodingNetwork/dense_3/kernel
?
KActorDistributionNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpReadVariableOp7ActorDistributionNetwork/EncodingNetwork/dense_3/kernel* 
_output_shapes
:
??*
dtype0
?
5ActorDistributionNetwork/EncodingNetwork/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense_3/bias
?
IActorDistributionNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense_3/bias*
_output_shapes	
:?*
dtype0
?
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*]
shared_nameNLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel
?
`ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel/Read/ReadVariableOpReadVariableOpLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel*
_output_shapes
:	?
*
dtype0
?
JActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*[
shared_nameLJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias
?
^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias/Read/ReadVariableOpReadVariableOpJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
1
2
	3

4
5

0
 
yw
VARIABLE_VALUE7ActorDistributionNetwork/EncodingNetwork/dense_2/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense_2/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE7ActorDistributionNetwork/EncodingNetwork/dense_3/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense_3/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE

ref
1
z
_encoder
_projection_networks
trainable_variables
regularization_losses
	variables
	keras_api
n
_postprocessing_layers
trainable_variables
regularization_losses
	variables
	keras_api
i
_projection_layer
trainable_variables
regularization_losses
	variables
	keras_api
*
0
1
2
	3

4
5
 
*
0
1
2
	3

4
5
?
non_trainable_variables
trainable_variables
layer_regularization_losses
 layer_metrics
regularization_losses

!layers
	variables
"metrics

#0
$1
%2

0
1
2
	3
 

0
1
2
	3
?
&non_trainable_variables
trainable_variables
'layer_regularization_losses
(layer_metrics
regularization_losses

)layers
	variables
*metrics
h


kernel
bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api


0
1
 


0
1
?
/non_trainable_variables
trainable_variables
0layer_regularization_losses
1layer_metrics
regularization_losses

2layers
	variables
3metrics
 
 
 

0
1
 
R
4trainable_variables
5regularization_losses
6	variables
7	keras_api
h

kernel
bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
h

kernel
	bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
 
 
 

#0
$1
%2
 


0
1
 


0
1
?
@non_trainable_variables
+trainable_variables
Alayer_regularization_losses
Blayer_metrics
,regularization_losses

Clayers
-	variables
Dmetrics
 
 
 

0
 
 
 
 
?
Enon_trainable_variables
4trainable_variables
Flayer_regularization_losses
Glayer_metrics
5regularization_losses

Hlayers
6	variables
Imetrics

0
1
 

0
1
?
Jnon_trainable_variables
8trainable_variables
Klayer_regularization_losses
Llayer_metrics
9regularization_losses

Mlayers
:	variables
Nmetrics

0
	1
 

0
	1
?
Onon_trainable_variables
<trainable_variables
Player_regularization_losses
Qlayer_metrics
=regularization_losses

Rlayers
>	variables
Smetrics
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
w
action_0/observationPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
j
action_0/rewardPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
m
action_0/step_typePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type7ActorDistributionNetwork/EncodingNetwork/dense_2/kernel5ActorDistributionNetwork/EncodingNetwork/dense_2/bias7ActorDistributionNetwork/EncodingNetwork/dense_3/kernel5ActorDistributionNetwork/EncodingNetwork/dense_3/biasLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernelJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3091554
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3091566
?
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3091588
?
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3091581
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpKActorDistributionNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpKActorDistributionNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOp`ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel/Read/ReadVariableOp^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias/Read/ReadVariableOpConst*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_3091957
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable7ActorDistributionNetwork/EncodingNetwork/dense_2/kernel5ActorDistributionNetwork/EncodingNetwork/dense_2/bias7ActorDistributionNetwork/EncodingNetwork/dense_3/kernel5ActorDistributionNetwork/EncodingNetwork/dense_3/biasLActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernelJActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_3091988??
]

__inference_<lambda>_3091193*(
_construction_contextkEagerRuntime*
_input_shapes 
?
7
%__inference_signature_wrapper_3091566

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *4
f/R-
+__inference_function_with_signature_30915612
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
??	
?	
)__inference_polymorphic_action_fn_3091517
	time_step
time_step_1
time_step_2
time_step_3b
Oactordistributionnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:	?_
Pactordistributionnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource:	?c
Oactordistributionnetwork_encodingnetwork_dense_3_matmul_readvariableop_resource:
??_
Pactordistributionnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resource:	?w
dactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource:	?
s
eactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource:

identity??GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp?\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp?[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9ActorDistributionNetwork/EncodingNetwork/flatten_13/Const?
;ActorDistributionNetwork/EncodingNetwork/flatten_13/ReshapeReshapetime_step_3BActorDistributionNetwork/EncodingNetwork/flatten_13/Const:output:0*
T0*'
_output_shapes
:?????????2=
;ActorDistributionNetwork/EncodingNetwork/flatten_13/Reshape?
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_2/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/flatten_13/Reshape:output:0NActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_2/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_2/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_2/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_2/Relu?
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_3/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_2/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_3/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_3/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_3/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_3/Relu?
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpReadVariableOpdactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02]
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp?
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_3/Relu:activations:0cActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2N
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul?
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOpReadVariableOpeactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02^
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp?
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddBiasAddVActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul:product:0dActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2O
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd?
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2F
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim?
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitSplitMActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim:output:0VActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????:?????????*
	num_split2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/split?
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape?
<ActorDistributionNetwork/TanhNormalProjectionNetwork/ReshapeReshapeCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:0KActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2>
<ActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape?
8ActorDistributionNetwork/TanhNormalProjectionNetwork/ExpExpCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:1*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp?
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const?
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2s
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2?	
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice?
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeEActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*
_output_shapes
:2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape?
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2{
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack?
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1?
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2?
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicetActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice?
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0|ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs?
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Y
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const?
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosFillxActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0`ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2S
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros?
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shape?
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zero?
@ActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shiftConst*
_output_shapes
: *
dtype0*
valueB
 *    2B
@ActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shift?
@ActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2B
@ActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scale?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zero?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShapeZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgsBroadcastArgs?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs:r0:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*'
_output_shapes
:?????????*
dtype02?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulMul?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normalAdd?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul:z:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mulMul?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal:z:0YActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/addAddV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mul:z:0ZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1Shape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2?&
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1ConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/perm?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose	Transpose?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2?"
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/addAddV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0EActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/TanhTanh?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanh?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/IdentityIdentity?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanh:y:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identity?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mulMul?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identity:output:0IActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scale:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addAddV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mul:z:0IActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shift:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?	
clip_by_value/MinimumMinimum?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/add:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:0H^ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp]^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:?????????:?????????:?????????:?????????: : : : : : 2?
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp2?
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp2?
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:RN
'
_output_shapes
:?????????
#
_user_specified_name	time_step
??	
?	
)__inference_polymorphic_action_fn_3091837
time_step_step_type
time_step_reward
time_step_discount
time_step_observationb
Oactordistributionnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:	?_
Pactordistributionnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource:	?c
Oactordistributionnetwork_encodingnetwork_dense_3_matmul_readvariableop_resource:
??_
Pactordistributionnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resource:	?w
dactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource:	?
s
eactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource:

identity??GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp?\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp?[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9ActorDistributionNetwork/EncodingNetwork/flatten_13/Const?
;ActorDistributionNetwork/EncodingNetwork/flatten_13/ReshapeReshapetime_step_observationBActorDistributionNetwork/EncodingNetwork/flatten_13/Const:output:0*
T0*'
_output_shapes
:?????????2=
;ActorDistributionNetwork/EncodingNetwork/flatten_13/Reshape?
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_2/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/flatten_13/Reshape:output:0NActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_2/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_2/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_2/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_2/Relu?
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_3/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_2/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_3/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_3/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_3/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_3/Relu?
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpReadVariableOpdactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02]
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp?
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_3/Relu:activations:0cActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2N
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul?
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOpReadVariableOpeactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02^
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp?
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddBiasAddVActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul:product:0dActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2O
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd?
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2F
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim?
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitSplitMActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim:output:0VActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????:?????????*
	num_split2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/split?
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape?
<ActorDistributionNetwork/TanhNormalProjectionNetwork/ReshapeReshapeCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:0KActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2>
<ActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape?
8ActorDistributionNetwork/TanhNormalProjectionNetwork/ExpExpCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:1*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp?
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const?
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2s
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2?	
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice?
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeEActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*
_output_shapes
:2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape?
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2{
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack?
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1?
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2?
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicetActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice?
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0|ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs?
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Y
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const?
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosFillxActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0`ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2S
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros?
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shape?
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zero?
@ActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shiftConst*
_output_shapes
: *
dtype0*
valueB
 *    2B
@ActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shift?
@ActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2B
@ActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scale?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zero?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShapeZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgsBroadcastArgs?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs:r0:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*'
_output_shapes
:?????????*
dtype02?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulMul?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normalAdd?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul:z:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mulMul?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal:z:0YActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/addAddV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mul:z:0ZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1Shape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2?&
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1ConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/perm?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose	Transpose?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2?"
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/addAddV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0EActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/TanhTanh?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanh?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/IdentityIdentity?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanh:y:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identity?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mulMul?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identity:output:0IActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scale:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addAddV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mul:z:0IActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shift:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?	
clip_by_value/MinimumMinimum?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/add:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:0H^ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp]^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:?????????:?????????:?????????:?????????: : : : : : 2?
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp2?
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp2?
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:X T
#
_output_shapes
:?????????
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:?????????
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:?????????
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:?????????
/
_user_specified_nametime_step/observation
??
?	
/__inference_polymorphic_distribution_fn_3091905
	step_type

reward
discount
observationb
Oactordistributionnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:	?_
Pactordistributionnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource:	?c
Oactordistributionnetwork_encodingnetwork_dense_3_matmul_readvariableop_resource:
??_
Pactordistributionnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resource:	?w
dactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource:	?
s
eactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource:
??GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp?\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp?[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp?StatefulPartitionedCall?
9ActorDistributionNetwork/EncodingNetwork/flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9ActorDistributionNetwork/EncodingNetwork/flatten_13/Const?
;ActorDistributionNetwork/EncodingNetwork/flatten_13/ReshapeReshapeobservationBActorDistributionNetwork/EncodingNetwork/flatten_13/Const:output:0*
T0*'
_output_shapes
:?????????2=
;ActorDistributionNetwork/EncodingNetwork/flatten_13/Reshape?
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_2/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/flatten_13/Reshape:output:0NActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_2/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_2/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_2/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_2/Relu?
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_3/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_2/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_3/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_3/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_3/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_3/Relu?
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpReadVariableOpdactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02]
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp?
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_3/Relu:activations:0cActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2N
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul?
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOpReadVariableOpeactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02^
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp?
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddBiasAddVActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul:product:0dActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2O
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd?
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2F
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim?
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitSplitMActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim:output:0VActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????:?????????*
	num_split2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/split?
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape?
<ActorDistributionNetwork/TanhNormalProjectionNetwork/ReshapeReshapeCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:0KActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2>
<ActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape?
8ActorDistributionNetwork/TanhNormalProjectionNetwork/ExpExpCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:1*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp?
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const?
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2s
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2?	
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice?
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeEActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*
_output_shapes
:2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape?
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2{
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack?
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1?
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2?
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicetActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice?
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0|ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs?
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Y
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const?
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosFillxActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0`ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2S
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros?
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shape?
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zero?
@ActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shiftConst*
_output_shapes
: *
dtype0*
valueB
 *    2B
@ActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shift?
@ActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2B
@ActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scale?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zero?
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference__raise_30919042
StatefulPartitionedCall*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:?????????:?????????:?????????:?????????: : : : : : 2?
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp2?
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp2?
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:TP
'
_output_shapes
:?????????
%
_user_specified_nameobservation
?
=
+__inference_function_with_signature_3091561

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_get_initial_state_30915602
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
/
__inference__raise_3091904??Assert/Assert?
Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Unable to make a CompositeTensor for "<tensorflow.python.ops.linalg.linear_operator_diag.LinearOperatorDiag object at 0x7f5074df7550>" of type `<class 'tensorflow.python.ops.linalg.linear_operator_diag.LinearOperatorDiag'>`. Email `tfprobability@tensorflow.org` or file an issue on github if you would benefit from this working. (Unable to serialize: No encoder for object [Tensor("ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:0", shape=(None, 5), dtype=float32)] of type [<class 'tensorflow.python.framework.ops.Tensor'>].)2
Assert/Constt
Assert/Assert/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
Assert/Assert/condition?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Unable to make a CompositeTensor for "<tensorflow.python.ops.linalg.linear_operator_diag.LinearOperatorDiag object at 0x7f5074df7550>" of type `<class 'tensorflow.python.ops.linalg.linear_operator_diag.LinearOperatorDiag'>`. Email `tfprobability@tensorflow.org` or file an issue on github if you would benefit from this working. (Unable to serialize: No encoder for object [Tensor("ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:0", shape=(None, 5), dtype=float32)] of type [<class 'tensorflow.python.framework.ops.Tensor'>].)2
Assert/Assert/data_0?
Assert/AssertAssert Assert/Assert/condition:output:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assert*(
_construction_contextkEagerRuntime*
_input_shapes 2
Assert/AssertAssert/Assert
?
'
%__inference_signature_wrapper_3091588?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *4
f/R-
+__inference_function_with_signature_30915842
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
?
k
+__inference_function_with_signature_3091573
unknown:	 
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference_<lambda>_30911902
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
?
?
%__inference_signature_wrapper_3091554
discount
observation

reward
	step_type
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?

	unknown_4:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *4
f/R-
+__inference_function_with_signature_30915322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:?????????:?????????:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:VR
'
_output_shapes
:?????????
'
_user_specified_name0/observation:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:PL
#
_output_shapes
:?????????
%
_user_specified_name0/step_type
?&
?
#__inference__traced_restore_3091988
file_prefix#
assignvariableop_variable:	 ]
Jassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_2_kernel:	?W
Hassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_2_bias:	?^
Jassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_3_kernel:
??W
Hassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_3_bias:	?r
_assignvariableop_5_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernel:	?
k
]assignvariableop_6_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_bias:


identity_8??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpJassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpHassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_2_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpJassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_3_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpHassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_3_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp_assignvariableop_5_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp]assignvariableop_6_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_7?

Identity_8IdentityIdentity_7:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*
T0*
_output_shapes
: 2

Identity_8"!

identity_8Identity_8:output:0*#
_input_shapes
: : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??	
?	
)__inference_polymorphic_action_fn_3091713
	step_type

reward
discount
observationb
Oactordistributionnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource:	?_
Pactordistributionnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource:	?c
Oactordistributionnetwork_encodingnetwork_dense_3_matmul_readvariableop_resource:
??_
Pactordistributionnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resource:	?w
dactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource:	?
s
eactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource:

identity??GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp?\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp?[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9ActorDistributionNetwork/EncodingNetwork/flatten_13/Const?
;ActorDistributionNetwork/EncodingNetwork/flatten_13/ReshapeReshapeobservationBActorDistributionNetwork/EncodingNetwork/flatten_13/Const:output:0*
T0*'
_output_shapes
:?????????2=
;ActorDistributionNetwork/EncodingNetwork/flatten_13/Reshape?
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_2/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/flatten_13/Reshape:output:0NActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_2/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_2/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_2/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_2/Relu?
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_3/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_2/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_3/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_3/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_3/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_3/Relu?
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOpReadVariableOpdactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02]
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp?
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_3/Relu:activations:0cActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2N
LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul?
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOpReadVariableOpeactordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02^
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp?
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAddBiasAddVActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul:product:0dActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2O
MActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd?
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2F
DActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim?
:ActorDistributionNetwork/TanhNormalProjectionNetwork/splitSplitMActorDistributionNetwork/TanhNormalProjectionNetwork/split/split_dim:output:0VActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd:output:0*
T0*:
_output_shapes(
&:?????????:?????????*
	num_split2<
:ActorDistributionNetwork/TanhNormalProjectionNetwork/split?
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
BActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape?
<ActorDistributionNetwork/TanhNormalProjectionNetwork/ReshapeReshapeCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:0KActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2>
<ActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape?
8ActorDistributionNetwork/TanhNormalProjectionNetwork/ExpExpCActorDistributionNetwork/TanhNormalProjectionNetwork/split:output:1*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp?
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const?
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2s
qActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2?	
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice?
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShapeEActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*
_output_shapes
:2m
kActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape?
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2{
yActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack?
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1?
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2}
{ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2?
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicetActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice?
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0|ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2u
sActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs?
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Y
WActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const?
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zerosFillxActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0`ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2S
QActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros?
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shape?
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2R
PActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zero?
@ActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shiftConst*
_output_shapes
: *
dtype0*
valueB
 *    2B
@ActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shift?
@ActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2B
@ActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scale?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/TanhNormalProjectionNetwork/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/zero?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShapeZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgsBroadcastArgs?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/values_0:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/BroadcastArgs:r0:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*'
_output_shapes
:?????????*
dtype02?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulMul?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normalAdd?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul:z:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mulMul?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/normal/random_normal:z:0YActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/ones:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/addAddV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/mul:z:0ZActorDistributionNetwork/TanhNormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1Shape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2?&
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1ConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/add:z:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat_1:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/perm?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose	Transpose?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ShapeShape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2?"
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/transpose:y:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1?	
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_Normal/sample/Reshape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul<ActorDistributionNetwork/TanhNormalProjectionNetwork/Exp:y:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/addAddV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0EActorDistributionNetwork/TanhNormalProjectionNetwork/Reshape:output:0*
T0*+
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ShapeShape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_sliceStridedSlice?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concatConcatV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/sample_shape:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/strided_slice:output:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ReshapeReshape?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_scale_matvec_linear_operator/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag_shift/forward/add:z:0?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/TanhTanh?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanh?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/IdentityIdentity?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/Tanh:y:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identity?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mulMul?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/Identity:output:0IActorDistributionNetwork/TanhNormalProjectionNetwork/scale/scale:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mul?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addAddV2?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/mul:z:0IActorDistributionNetwork/TanhNormalProjectionNetwork/shift/shift:output:0*
T0*'
_output_shapes
:?????????2?
?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/addw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?	
clip_by_value/MinimumMinimum?ActorDistributionNetwork_TanhNormalProjectionNetwork_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanhActorDistributionNetwork_TanhNormalProjectionNetwork_MultivariateNormalDiag/sample/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_tanh/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_chain_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_shift_of_ActorDistributionNetwork_TanhNormalProjectionNetwork_scale/forward/ActorDistributionNetwork_TanhNormalProjectionNetwork_shift/forward/add:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:0H^ActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp]^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\^ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:?????????:?????????:?????????:?????????: : : : : : 2?
GActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_2/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_2/MatMul/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp2?
\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp\ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/BiasAdd/ReadVariableOp2?
[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp[ActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:TP
'
_output_shapes
:?????????
%
_user_specified_nameobservation
?
?
+__inference_function_with_signature_3091532
	step_type

reward
discount
observation
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?

	unknown_4:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_polymorphic_action_fn_30915172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:?????????:?????????:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:?????????
%
_user_specified_name0/step_type:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:OK
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:VR
'
_output_shapes
:?????????
'
_user_specified_name0/observation
?
-
+__inference_function_with_signature_3091584?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference_<lambda>_30911932
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
?
e
%__inference_signature_wrapper_3091581
unknown:	 
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *4
f/R-
+__inference_function_with_signature_30915732
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
?
7
%__inference_get_initial_state_3091560

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
? 
?
 __inference__traced_save_3091957
file_prefix'
#savev2_variable_read_readvariableop	V
Rsavev2_actordistributionnetwork_encodingnetwork_dense_2_kernel_read_readvariableopT
Psavev2_actordistributionnetwork_encodingnetwork_dense_2_bias_read_readvariableopV
Rsavev2_actordistributionnetwork_encodingnetwork_dense_3_kernel_read_readvariableopT
Psavev2_actordistributionnetwork_encodingnetwork_dense_3_bias_read_readvariableopk
gsavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernel_read_readvariableopi
esavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopRsavev2_actordistributionnetwork_encodingnetwork_dense_2_kernel_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_2_bias_read_readvariableopRsavev2_actordistributionnetwork_encodingnetwork_dense_3_kernel_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_3_bias_read_readvariableopgsavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_kernel_read_readvariableopesavev2_actordistributionnetwork_tanhnormalprojectionnetwork_projection_layer_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*O
_input_shapes>
<: : :	?:?:
??:?:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :%!

_output_shapes
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: 
?
7
%__inference_get_initial_state_3091908

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
c
__inference_<lambda>_3091190!
readvariableop_resource:	 
identity	??ReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpj
IdentityIdentityReadVariableOp:value:0^ReadVariableOp*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
action?
4

0/discount&
action_0/discount:0?????????
>
0/observation-
action_0/observation:0?????????
0
0/reward$
action_0/reward:0?????????
6
0/step_type'
action_0/step_type:0?????????:
action0
StatefulPartitionedCall:0?????????tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:??
?

train_step
metadata
model_variables
_all_assets

signatures

Taction
Udistribution
Vget_initial_state
Wget_metadata
Xget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
K
0
1
2
	3

4
5"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
`

Yaction
Zget_initial_state
[get_train_step
\get_metadata"
signature_map
J:H	?27ActorDistributionNetwork/EncodingNetwork/dense_2/kernel
D:B?25ActorDistributionNetwork/EncodingNetwork/dense_2/bias
K:I
??27ActorDistributionNetwork/EncodingNetwork/dense_3/kernel
D:B?25ActorDistributionNetwork/EncodingNetwork/dense_3/bias
_:]	?
2LActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/kernel
X:V
2JActorDistributionNetwork/TanhNormalProjectionNetwork/projection_layer/bias
1
ref
1"
trackable_tuple_wrapper
?
_encoder
_projection_networks
trainable_variables
regularization_losses
	variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "ActorDistributionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ActorDistributionNetwork", "config": {"layer was saved without config": true}}
?
_postprocessing_layers
trainable_variables
regularization_losses
	variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EncodingNetwork", "config": {"layer was saved without config": true}}
?
_projection_layer
trainable_variables
regularization_losses
	variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "TanhNormalProjectionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "TanhNormalProjectionNetwork", "config": {"layer was saved without config": true}}
J
0
1
2
	3

4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
	3

4
5"
trackable_list_wrapper
?
non_trainable_variables
trainable_variables
layer_regularization_losses
 layer_metrics
regularization_losses

!layers
	variables
"metrics
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
5
#0
$1
%2"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
?
&non_trainable_variables
trainable_variables
'layer_regularization_losses
(layer_metrics
regularization_losses

)layers
	variables
*metrics
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?


kernel
bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
c__call__
*d&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "projection_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "projection_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
/non_trainable_variables
trainable_variables
0layer_regularization_losses
1layer_metrics
regularization_losses

2layers
	variables
3metrics
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4trainable_variables
5regularization_losses
6	variables
7	keras_api
e__call__
*f&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
g__call__
*h&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 5]}}
?

kernel
	bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
i__call__
*j&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
#0
$1
%2"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
@non_trainable_variables
+trainable_variables
Alayer_regularization_losses
Blayer_metrics
,regularization_losses

Clayers
-	variables
Dmetrics
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables
4trainable_variables
Flayer_regularization_losses
Glayer_metrics
5regularization_losses

Hlayers
6	variables
Imetrics
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Jnon_trainable_variables
8trainable_variables
Klayer_regularization_losses
Llayer_metrics
9regularization_losses

Mlayers
:	variables
Nmetrics
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
?
Onon_trainable_variables
<trainable_variables
Player_regularization_losses
Qlayer_metrics
=regularization_losses

Rlayers
>	variables
Smetrics
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
)__inference_polymorphic_action_fn_3091713
)__inference_polymorphic_action_fn_3091837?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_polymorphic_distribution_fn_3091905?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_get_initial_state_3091908?
???
FullArgSpec!
args?
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_3091193"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_3091190"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_3091554
0/discount0/observation0/reward0/step_type"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_3091566
batch_size"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_3091581"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_3091588"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpecU
argsM?J
jself
jobservations
j	step_type
jnetwork_state

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecU
argsM?J
jself
jobservations
j	step_type
jnetwork_state

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec?
args7?4
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec?
args7?4
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ;
__inference_<lambda>_3091190?

? 
? "? 	4
__inference_<lambda>_3091193?

? 
? "? R
%__inference_get_initial_state_3091908)"?
?
?

batch_size 
? "? ?
)__inference_polymorphic_action_fn_3091713?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????4
observation%?"
observation?????????
? 
? "V?S

PolicyStep*
action ?
action?????????
state? 
info? ?
)__inference_polymorphic_action_fn_3091837?	
???
???
???
TimeStep6
	step_type)?&
time_step/step_type?????????0
reward&?#
time_step/reward?????????4
discount(?%
time_step/discount?????????>
observation/?,
time_step/observation?????????
? 
? "V?S

PolicyStep*
action ?
action?????????
state? 
info? ?
/__inference_polymorphic_distribution_fn_3091905?	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????4
observation%?"
observation?????????
? 
? "
 ?
%__inference_signature_wrapper_3091554?	
???
? 
???
.

0/discount ?

0/discount?????????
8
0/observation'?$
0/observation?????????
*
0/reward?
0/reward?????????
0
0/step_type!?
0/step_type?????????"/?,
*
action ?
action?????????`
%__inference_signature_wrapper_309156670?-
? 
&?#
!

batch_size?

batch_size "? Y
%__inference_signature_wrapper_30915810?

? 
? "?

int64?
int64 	=
%__inference_signature_wrapper_3091588?

? 
? "? 