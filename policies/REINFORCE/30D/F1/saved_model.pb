??
??
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
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
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
@
Softplus
features"T
activations"T"
Ttype:
2
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.5.02v2.5.0-0-ga4dfb8d1a718܋
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
5ActorDistributionNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense/kernel
?
IActorDistributionNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	?*
dtype0
?
3ActorDistributionNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*D
shared_name53ActorDistributionNetwork/EncodingNetwork/dense/bias
?
GActorDistributionNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp3ActorDistributionNetwork/EncodingNetwork/dense/bias*
_output_shapes	
:?*
dtype0
?
7ActorDistributionNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*H
shared_name97ActorDistributionNetwork/EncodingNetwork/dense_1/kernel
?
KActorDistributionNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel* 
_output_shapes
:
??*
dtype0
?
5ActorDistributionNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense_1/bias
?
IActorDistributionNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense_1/bias*
_output_shapes	
:?*
dtype0
?
@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias
?
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias/Read/ReadVariableOpReadVariableOp@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias*
_output_shapes
:*
dtype0
?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*_
shared_namePNActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel
?
bActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel/Read/ReadVariableOpReadVariableOpNActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel*
_output_shapes
:	?*
dtype0
?
LActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*]
shared_nameNLActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias
?
`ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias/Read/ReadVariableOpReadVariableOpLActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
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
1
0
1
2
	3

4
5
6

0
 
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE3ActorDistributionNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUENActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE

ref
1

_actor_network
z
_encoder
_projection_networks
regularization_losses
	variables
trainable_variables
	keras_api
n
_postprocessing_layers
regularization_losses
	variables
trainable_variables
	keras_api
z
_means_projection_layer
	_bias
regularization_losses
	variables
trainable_variables
 	keras_api
 
1
0
1
2
	3
4
5

6
1
0
1
2
	3
4
5

6
?
regularization_losses
!metrics

"layers
#layer_metrics
	variables
$layer_regularization_losses
%non_trainable_variables
trainable_variables

&0
'1
(2
 

0
1
2
	3

0
1
2
	3
?
regularization_losses
)metrics

*layers
+layer_metrics
	variables
,layer_regularization_losses
-non_trainable_variables
trainable_variables
h

kernel
bias
.regularization_losses
/	variables
0trainable_variables
1	keras_api
\

bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
 

0
1

2

0
1

2
?
regularization_losses
6metrics

7layers
8layer_metrics
	variables
9layer_regularization_losses
:non_trainable_variables
trainable_variables
 

0
1
 
 
 
R
;regularization_losses
<	variables
=trainable_variables
>	keras_api
h

kernel
bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
h

kernel
	bias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
 

&0
'1
(2
 
 
 
 

0
1

0
1
?
.regularization_losses
Gmetrics

Hlayers
Ilayer_metrics
/	variables
Jlayer_regularization_losses
Knon_trainable_variables
0trainable_variables
 


0


0
?
2regularization_losses
Lmetrics

Mlayers
Nlayer_metrics
3	variables
Olayer_regularization_losses
Pnon_trainable_variables
4trainable_variables
 

0
1
 
 
 
 
 
 
?
;regularization_losses
Qmetrics

Rlayers
Slayer_metrics
<	variables
Tlayer_regularization_losses
Unon_trainable_variables
=trainable_variables
 

0
1

0
1
?
?regularization_losses
Vmetrics

Wlayers
Xlayer_metrics
@	variables
Ylayer_regularization_losses
Znon_trainable_variables
Atrainable_variables
 

0
	1

0
	1
?
Cregularization_losses
[metrics

\layers
]layer_metrics
D	variables
^layer_regularization_losses
_non_trainable_variables
Etrainable_variables
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
:?????????*
dtype0*
shape:?????????
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
?
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type5ActorDistributionNetwork/EncodingNetwork/dense/kernel3ActorDistributionNetwork/EncodingNetwork/dense/bias7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel5ActorDistributionNetwork/EncodingNetwork/dense_1/biasNActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernelLActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1660294
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
%__inference_signature_wrapper_1660306
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
%__inference_signature_wrapper_1660328
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
%__inference_signature_wrapper_1660321
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpKActorDistributionNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpTActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias/Read/ReadVariableOpbActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel/Read/ReadVariableOp`ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias/Read/ReadVariableOpConst*
Tin
2
	*
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
 __inference__traced_save_1660682
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable5ActorDistributionNetwork/EncodingNetwork/dense/kernel3ActorDistributionNetwork/EncodingNetwork/dense/bias7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel5ActorDistributionNetwork/EncodingNetwork/dense_1/bias@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/biasNActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernelLActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias*
Tin
2	*
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
#__inference__traced_restore_1660716??

?
?
+__inference_function_with_signature_1660270
	step_type

reward
discount
observation
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *2
f-R+
)__inference_polymorphic_action_fn_16602532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:?????????: : : : : : : 22
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
:?????????
'
_user_specified_name0/observation
?
7
%__inference_get_initial_state_1660300

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
?
-
+__inference_function_with_signature_1660324?
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
__inference_<lambda>_16599472
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
)__inference_polymorphic_action_fn_1660434
	step_type

reward
discount
observation`
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	?]
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	?c
Oactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
??_
Pactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	?y
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:	?u
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:i
[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity??EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   28
6ActorDistributionNetwork/EncodingNetwork/flatten/Const?
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/EncodingNetwork/flatten/Reshape?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense/MatMul?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02G
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd?
3ActorDistributionNetwork/EncodingNetwork/dense/ReluRelu?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????25
3ActorDistributionNetwork/EncodingNetwork/dense/Relu?
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/dense/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_1/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_1/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_1/Relu?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02_
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_1/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2P
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpgactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02`
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAddXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0fActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2Q
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd?
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape?
8ActorDistributionNetwork/NormalProjectionNetwork/ReshapeReshapeXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0GActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/NormalProjectionNetwork/Reshape?
5ActorDistributionNetwork/NormalProjectionNetwork/TanhTanhAActorDistributionNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:?????????27
5ActorDistributionNetwork/NormalProjectionNetwork/Tanh?
6ActorDistributionNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??28
6ActorDistributionNetwork/NormalProjectionNetwork/mul/x?
4ActorDistributionNetwork/NormalProjectionNetwork/mulMul?ActorDistributionNetwork/NormalProjectionNetwork/mul/x:output:09ActorDistributionNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:?????????26
4ActorDistributionNetwork/NormalProjectionNetwork/mul?
6ActorDistributionNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6ActorDistributionNetwork/NormalProjectionNetwork/add/x?
4ActorDistributionNetwork/NormalProjectionNetwork/addAddV2?ActorDistributionNetwork/NormalProjectionNetwork/add/x:output:08ActorDistributionNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:?????????26
4ActorDistributionNetwork/NormalProjectionNetwork/add?
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like	ZerosLike8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:?????????2=
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02T
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAdd?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2E
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape?
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeLActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2<
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1?
9ActorDistributionNetwork/NormalProjectionNetwork/SoftplusSoftplusCActorDistributionNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2;
9ActorDistributionNetwork/NormalProjectionNetwork/Softplus?
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2i
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const?
mActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeGActorDistributionNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2?	
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice?
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2i
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape?
uActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2w
uActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack?
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2y
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1?
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2y
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2?
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicepActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0~ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2q
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice?
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0xActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2q
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs?
SActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2U
SActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const?
MActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFilltActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0\ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2O
MActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros?
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2N
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/ones?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shape?
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2N
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zero?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeVActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1:output:0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape?	
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape?
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis?
SActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0aActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat?
UActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/IdentityIdentity8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:?????????2W
UActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity?
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Z
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const?
RActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosFill\ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0aActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2T
RActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros?
PActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addAddV2^ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity:output:0[ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros:output:0*
T0*'
_output_shapes
:?????????2R
PActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addm
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeTActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToTActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0&Deterministic_1/sample/concat:output:0*
T0*+
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
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
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:0F^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpS^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:?????????: : : : : : : 2?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
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
:?????????
%
_user_specified_nameobservation
??
?
/__inference_polymorphic_distribution_fn_1660627
	step_type

reward
discount
observation`
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	?]
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	?c
Oactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
??_
Pactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	?y
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:	?u
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:i
[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity??EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   28
6ActorDistributionNetwork/EncodingNetwork/flatten/Const?
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshapeobservation?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/EncodingNetwork/flatten/Reshape?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense/MatMul?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02G
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd?
3ActorDistributionNetwork/EncodingNetwork/dense/ReluRelu?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????25
3ActorDistributionNetwork/EncodingNetwork/dense/Relu?
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/dense/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_1/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_1/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_1/Relu?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02_
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_1/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2P
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpgactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02`
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAddXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0fActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2Q
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd?
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape?
8ActorDistributionNetwork/NormalProjectionNetwork/ReshapeReshapeXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0GActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/NormalProjectionNetwork/Reshape?
5ActorDistributionNetwork/NormalProjectionNetwork/TanhTanhAActorDistributionNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:?????????27
5ActorDistributionNetwork/NormalProjectionNetwork/Tanh?
6ActorDistributionNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??28
6ActorDistributionNetwork/NormalProjectionNetwork/mul/x?
4ActorDistributionNetwork/NormalProjectionNetwork/mulMul?ActorDistributionNetwork/NormalProjectionNetwork/mul/x:output:09ActorDistributionNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:?????????26
4ActorDistributionNetwork/NormalProjectionNetwork/mul?
6ActorDistributionNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6ActorDistributionNetwork/NormalProjectionNetwork/add/x?
4ActorDistributionNetwork/NormalProjectionNetwork/addAddV2?ActorDistributionNetwork/NormalProjectionNetwork/add/x:output:08ActorDistributionNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:?????????26
4ActorDistributionNetwork/NormalProjectionNetwork/add?
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like	ZerosLike8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:?????????2=
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02T
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAdd?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2E
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape?
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeLActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2<
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1?
9ActorDistributionNetwork/NormalProjectionNetwork/SoftplusSoftplusCActorDistributionNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2;
9ActorDistributionNetwork/NormalProjectionNetwork/Softplus?
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2i
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const?
mActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeGActorDistributionNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2?	
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice?
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2i
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape?
uActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2w
uActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack?
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2y
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1?
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2y
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2?
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicepActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0~ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2q
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice?
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0xActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2q
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs?
SActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2U
SActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const?
MActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFilltActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0\ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2O
MActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros?
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2N
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/ones?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shape?
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2N
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zero?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeVActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1:output:0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape?	
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape?
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis?
SActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0aActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat?
UActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/IdentityIdentity8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:?????????2W
UActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity?
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Z
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const?
RActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosFill\ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0aActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2T
RActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros?
PActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addAddV2^ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity:output:0[ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros:output:0*
T0*'
_output_shapes
:?????????2R
PActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addm
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtolq
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic_1/atolq
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic_1/rtol?
IdentityIdentityTActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0F^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpS^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identityq
Deterministic_2/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic_2/atolq
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic_2/rtol"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:?????????: : : : : : : 2?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
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
:?????????
%
_user_specified_nameobservation
?
k
+__inference_function_with_signature_1660313
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
__inference_<lambda>_16599442
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
?
7
%__inference_signature_wrapper_1660306

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
+__inference_function_with_signature_16603012
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
˲
?
)__inference_polymorphic_action_fn_1660253
	time_step
time_step_1
time_step_2
time_step_3`
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	?]
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	?c
Oactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
??_
Pactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	?y
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:	?u
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:i
[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity??EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   28
6ActorDistributionNetwork/EncodingNetwork/flatten/Const?
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/EncodingNetwork/flatten/Reshape?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense/MatMul?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02G
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd?
3ActorDistributionNetwork/EncodingNetwork/dense/ReluRelu?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????25
3ActorDistributionNetwork/EncodingNetwork/dense/Relu?
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/dense/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_1/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_1/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_1/Relu?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02_
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_1/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2P
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpgactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02`
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAddXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0fActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2Q
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd?
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape?
8ActorDistributionNetwork/NormalProjectionNetwork/ReshapeReshapeXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0GActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/NormalProjectionNetwork/Reshape?
5ActorDistributionNetwork/NormalProjectionNetwork/TanhTanhAActorDistributionNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:?????????27
5ActorDistributionNetwork/NormalProjectionNetwork/Tanh?
6ActorDistributionNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??28
6ActorDistributionNetwork/NormalProjectionNetwork/mul/x?
4ActorDistributionNetwork/NormalProjectionNetwork/mulMul?ActorDistributionNetwork/NormalProjectionNetwork/mul/x:output:09ActorDistributionNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:?????????26
4ActorDistributionNetwork/NormalProjectionNetwork/mul?
6ActorDistributionNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6ActorDistributionNetwork/NormalProjectionNetwork/add/x?
4ActorDistributionNetwork/NormalProjectionNetwork/addAddV2?ActorDistributionNetwork/NormalProjectionNetwork/add/x:output:08ActorDistributionNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:?????????26
4ActorDistributionNetwork/NormalProjectionNetwork/add?
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like	ZerosLike8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:?????????2=
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02T
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAdd?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2E
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape?
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeLActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2<
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1?
9ActorDistributionNetwork/NormalProjectionNetwork/SoftplusSoftplusCActorDistributionNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2;
9ActorDistributionNetwork/NormalProjectionNetwork/Softplus?
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2i
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const?
mActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeGActorDistributionNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2?	
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice?
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2i
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape?
uActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2w
uActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack?
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2y
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1?
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2y
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2?
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicepActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0~ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2q
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice?
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0xActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2q
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs?
SActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2U
SActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const?
MActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFilltActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0\ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2O
MActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros?
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2N
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/ones?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shape?
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2N
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zero?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeVActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1:output:0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape?	
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape?
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis?
SActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0aActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat?
UActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/IdentityIdentity8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:?????????2W
UActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity?
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Z
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const?
RActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosFill\ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0aActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2T
RActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros?
PActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addAddV2^ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity:output:0[ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros:output:0*
T0*'
_output_shapes
:?????????2R
PActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addm
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeTActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToTActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0&Deterministic_1/sample/concat:output:0*
T0*+
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
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
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:0F^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpS^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:?????????: : : : : : : 2?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:N J
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
:?????????
#
_user_specified_name	time_step
?
c
__inference_<lambda>_1659944!
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
ReadVariableOpReadVariableOp
?
e
%__inference_signature_wrapper_1660321
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
+__inference_function_with_signature_16603132
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
%__inference_signature_wrapper_1660294
discount
observation

reward
	step_type
unknown:	?
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
	unknown_3:	?
	unknown_4:
	unknown_5:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*)
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *4
f/R-
+__inference_function_with_signature_16602702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:?????????: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:VR
'
_output_shapes
:?????????
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
??
?
)__inference_polymorphic_action_fn_1660539
time_step_step_type
time_step_reward
time_step_discount
time_step_observation`
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	?]
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	?c
Oactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
??_
Pactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	?y
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:	?u
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:i
[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity??EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   28
6ActorDistributionNetwork/EncodingNetwork/flatten/Const?
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_observation?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/EncodingNetwork/flatten/Reshape?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02F
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp?
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense/MatMul?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02G
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp?
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd?
3ActorDistributionNetwork/EncodingNetwork/dense/ReluRelu?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????25
3ActorDistributionNetwork/EncodingNetwork/dense/Relu?
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp?
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/dense/Relu:activations:0NActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????29
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul?
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_1/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd?
5ActorDistributionNetwork/EncodingNetwork/dense_1/ReluReluAActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????27
5ActorDistributionNetwork/EncodingNetwork/dense_1/Relu?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02_
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/dense_1/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2P
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpgactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02`
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAddXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0fActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2Q
OActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd?
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2@
>ActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape?
8ActorDistributionNetwork/NormalProjectionNetwork/ReshapeReshapeXActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0GActorDistributionNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2:
8ActorDistributionNetwork/NormalProjectionNetwork/Reshape?
5ActorDistributionNetwork/NormalProjectionNetwork/TanhTanhAActorDistributionNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*'
_output_shapes
:?????????27
5ActorDistributionNetwork/NormalProjectionNetwork/Tanh?
6ActorDistributionNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??28
6ActorDistributionNetwork/NormalProjectionNetwork/mul/x?
4ActorDistributionNetwork/NormalProjectionNetwork/mulMul?ActorDistributionNetwork/NormalProjectionNetwork/mul/x:output:09ActorDistributionNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*'
_output_shapes
:?????????26
4ActorDistributionNetwork/NormalProjectionNetwork/mul?
6ActorDistributionNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6ActorDistributionNetwork/NormalProjectionNetwork/add/x?
4ActorDistributionNetwork/NormalProjectionNetwork/addAddV2?ActorDistributionNetwork/NormalProjectionNetwork/add/x:output:08ActorDistributionNetwork/NormalProjectionNetwork/mul:z:0*
T0*'
_output_shapes
:?????????26
4ActorDistributionNetwork/NormalProjectionNetwork/add?
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like	ZerosLike8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:?????????2=
;ActorDistributionNetwork/NormalProjectionNetwork/zeros_like?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp[actordistributionnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02T
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp?
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAdd?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0ZActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2E
CActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape?
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeLActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2<
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1?
9ActorDistributionNetwork/NormalProjectionNetwork/SoftplusSoftplusCActorDistributionNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2;
9ActorDistributionNetwork/NormalProjectionNetwork/Softplus?
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ConstConst*
_output_shapes
: *
dtype0*
value	B :2i
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Const?
mActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2o
mActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShapeGActorDistributionNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2?	
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSlice?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice?
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
:2i
gActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape?
uActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2w
uActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack?
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????2y
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1?
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2y
wActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2?
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSlicepActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0~ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2q
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice?
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgsBroadcastArgs?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0xActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2q
oActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs?
SActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2U
SActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const?
MActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zerosFilltActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/shapes_from_loc_and_scale/BroadcastArgs:r0:0\ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:?????????2O
MActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros?
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2N
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/ones?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/sample_shape?
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 2N
LActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zero?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ShapeShapeVActorDistributionNetwork/NormalProjectionNetwork/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/Shape:output:0?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs/s1:output:0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/BroadcastArgs:r0:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape?	
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_SampleActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_Normal/batch_shape_tensor/batch_shape:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/ConstConst*
_output_shapes
:*
dtype0*
valueB:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shapeIdentity?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/Const:output:0*
T0*
_output_shapes
:2?
?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape?
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis?
SActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concatConcatV2?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/batch_shape_tensor/batch_shape:output:0?ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag_1/event_shape_tensor/event_shape:output:0aActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat/axis:output:0*
N*
T0*
_output_shapes
:2U
SActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat?
UActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/IdentityIdentity8ActorDistributionNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:?????????2W
UActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity?
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2Z
XActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const?
RActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zerosFill\ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/concat:output:0aActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2T
RActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros?
PActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addAddV2^ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/Identity:output:0[ActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/zeros:output:0*
T0*'
_output_shapes
:?????????2R
PActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/addm
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/atolm
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShapeTActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastToTActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0&Deterministic_1/sample/concat:output:0*
T0*+
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapew
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
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
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:0F^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpS^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:?????????: : : : : : : 2?
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2?
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2?
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2?
RActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpRActorDistributionNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2?
^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:X T
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
:?????????
/
_user_specified_nametime_step/observation
]

__inference_<lambda>_1659947*(
_construction_contextkEagerRuntime*
_input_shapes 
?*
?
#__inference__traced_restore_1660716
file_prefix#
assignvariableop_variable:	 [
Hassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_kernel:	?U
Fassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_bias:	?^
Jassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_1_kernel:
??W
Hassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_1_bias:	?a
Sassignvariableop_5_actordistributionnetwork_normalprojectionnetwork_bias_layer_bias:t
aassignvariableop_6_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernel:	?m
_assignvariableop_7_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_bias:

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		2
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
AssignVariableOp_1AssignVariableOpHassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpFassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpJassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpHassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpSassignvariableop_5_actordistributionnetwork_normalprojectionnetwork_bias_layer_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpaassignvariableop_6_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp_assignvariableop_7_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
7
%__inference_get_initial_state_1660630

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
=
+__inference_function_with_signature_1660301

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
%__inference_get_initial_state_16603002
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
?
'
%__inference_signature_wrapper_1660328?
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
+__inference_function_with_signature_16603242
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
?#
?
 __inference__traced_save_1660682
file_prefix'
#savev2_variable_read_readvariableop	T
Psavev2_actordistributionnetwork_encodingnetwork_dense_kernel_read_readvariableopR
Nsavev2_actordistributionnetwork_encodingnetwork_dense_bias_read_readvariableopV
Rsavev2_actordistributionnetwork_encodingnetwork_dense_1_kernel_read_readvariableopT
Psavev2_actordistributionnetwork_encodingnetwork_dense_1_bias_read_readvariableop_
[savev2_actordistributionnetwork_normalprojectionnetwork_bias_layer_bias_read_readvariableopm
isavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernel_read_readvariableopk
gsavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_kernel_read_readvariableopNsavev2_actordistributionnetwork_encodingnetwork_dense_bias_read_readvariableopRsavev2_actordistributionnetwork_encodingnetwork_dense_1_kernel_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_1_bias_read_readvariableop[savev2_actordistributionnetwork_normalprojectionnetwork_bias_layer_bias_read_readvariableopisavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernel_read_readvariableopgsavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2		2
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

identity_1Identity_1:output:0*U
_input_shapesD
B: : :	?:?:
??:?::	?:: 2(
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
:	?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::	

_output_shapes
: "?L
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
action_0/observation:0?????????
0
0/reward$
action_0/reward:0?????????
6
0/step_type'
action_0/step_type:0?????????:
action0
StatefulPartitionedCall:0?????????tensorflow/serving/predict*e
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

`action
adistribution
bget_initial_state
cget_metadata
dget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
R
0
1
2
	3

4
5
6"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
`

eaction
fget_initial_state
gget_train_step
hget_metadata"
signature_map
H:F	?25ActorDistributionNetwork/EncodingNetwork/dense/kernel
B:@?23ActorDistributionNetwork/EncodingNetwork/dense/bias
K:I
??27ActorDistributionNetwork/EncodingNetwork/dense_1/kernel
D:B?25ActorDistributionNetwork/EncodingNetwork/dense_1/bias
N:L2@ActorDistributionNetwork/NormalProjectionNetwork/bias_layer/bias
a:_	?2NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel
Z:X2LActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias
1
ref
1"
trackable_tuple_wrapper
2
_actor_network"
_generic_user_object
?
_encoder
_projection_networks
regularization_losses
	variables
trainable_variables
	keras_api
i__call__
*j&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "ActorDistributionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ActorDistributionNetwork", "config": {"layer was saved without config": true}}
?
_postprocessing_layers
regularization_losses
	variables
trainable_variables
	keras_api
k__call__
*l&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EncodingNetwork", "config": {"layer was saved without config": true}}
?
_means_projection_layer
	_bias
regularization_losses
	variables
trainable_variables
 	keras_api
m__call__
*n&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "NormalProjectionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "NormalProjectionNetwork", "config": {"layer was saved without config": true}}
 "
trackable_list_wrapper
Q
0
1
2
	3
4
5

6"
trackable_list_wrapper
Q
0
1
2
	3
4
5

6"
trackable_list_wrapper
?
regularization_losses
!metrics

"layers
#layer_metrics
	variables
$layer_regularization_losses
%non_trainable_variables
trainable_variables
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
?
regularization_losses
)metrics

*layers
+layer_metrics
	variables
,layer_regularization_losses
-non_trainable_variables
trainable_variables
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
?

kernel
bias
.regularization_losses
/	variables
0trainable_variables
1	keras_api
o__call__
*p&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "means_projection_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "means_projection_layer", "trainable": true, "dtype": "float32", "units": 30, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.1, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
?

bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
q__call__
*r&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "bias_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BiasLayer", "config": {"name": "bias_layer", "trainable": true, "dtype": "float32", "bias_initializer": {"class_name": "Constant", "config": {"value": -0.8697231582271624}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 30]}}
 "
trackable_list_wrapper
5
0
1

2"
trackable_list_wrapper
5
0
1

2"
trackable_list_wrapper
?
regularization_losses
6metrics

7layers
8layer_metrics
	variables
9layer_regularization_losses
:non_trainable_variables
trainable_variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;regularization_losses
<	variables
=trainable_variables
>	keras_api
s__call__
*t&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
u__call__
*v&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 30]}}
?

kernel
	bias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
w__call__
*x&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
.regularization_losses
Gmetrics

Hlayers
Ilayer_metrics
/	variables
Jlayer_regularization_losses
Knon_trainable_variables
0trainable_variables
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
'

0"
trackable_list_wrapper
?
2regularization_losses
Lmetrics

Mlayers
Nlayer_metrics
3	variables
Olayer_regularization_losses
Pnon_trainable_variables
4trainable_variables
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
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
trackable_list_wrapper
?
;regularization_losses
Qmetrics

Rlayers
Slayer_metrics
<	variables
Tlayer_regularization_losses
Unon_trainable_variables
=trainable_variables
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?regularization_losses
Vmetrics

Wlayers
Xlayer_metrics
@	variables
Ylayer_regularization_losses
Znon_trainable_variables
Atrainable_variables
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
?
Cregularization_losses
[metrics

\layers
]layer_metrics
D	variables
^layer_regularization_losses
_non_trainable_variables
Etrainable_variables
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
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
)__inference_polymorphic_action_fn_1660434
)__inference_polymorphic_action_fn_1660539?
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
/__inference_polymorphic_distribution_fn_1660627?
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
%__inference_get_initial_state_1660630?
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
__inference_<lambda>_1659947"?
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
__inference_<lambda>_1659944"?
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
%__inference_signature_wrapper_1660294
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
%__inference_signature_wrapper_1660306
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
%__inference_signature_wrapper_1660321"?
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
%__inference_signature_wrapper_1660328"?
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
__inference_<lambda>_1659944?

? 
? "? 	4
__inference_<lambda>_1659947?

? 
? "? R
%__inference_get_initial_state_1660630)"?
?
?

batch_size 
? "? ?
)__inference_polymorphic_action_fn_1660434?	
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
observation?????????
? 
? "V?S

PolicyStep*
action ?
action?????????
state? 
info? ?
)__inference_polymorphic_action_fn_1660539?	
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
time_step/observation?????????
? 
? "V?S

PolicyStep*
action ?
action?????????
state? 
info? ?
/__inference_polymorphic_distribution_fn_1660627?	
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
observation?????????
? 
? "???

PolicyStep?
action?????Ã??~
`
C?@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
.?+
)
loc"?
Identity?????????
? _TFPTypeSpec
state? 
info? ?
%__inference_signature_wrapper_1660294?	
???
? 
???
.

0/discount ?

0/discount?????????
8
0/observation'?$
0/observation?????????
*
0/reward?
0/reward?????????
0
0/step_type!?
0/step_type?????????"/?,
*
action ?
action?????????`
%__inference_signature_wrapper_166030670?-
? 
&?#
!

batch_size?

batch_size "? Y
%__inference_signature_wrapper_16603210?

? 
? "?

int64?
int64 	=
%__inference_signature_wrapper_1660328?

? 
? "? 