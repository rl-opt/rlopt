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
Ttype"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
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
8ActorDistributionNetwork/EncodingNetwork/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*I
shared_name:8ActorDistributionNetwork/EncodingNetwork/dense_10/kernel
?
LActorDistributionNetwork/EncodingNetwork/dense_10/kernel/Read/ReadVariableOpReadVariableOp8ActorDistributionNetwork/EncodingNetwork/dense_10/kernel*
_output_shapes
:	?*
dtype0
?
6ActorDistributionNetwork/EncodingNetwork/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*G
shared_name86ActorDistributionNetwork/EncodingNetwork/dense_10/bias
?
JActorDistributionNetwork/EncodingNetwork/dense_10/bias/Read/ReadVariableOpReadVariableOp6ActorDistributionNetwork/EncodingNetwork/dense_10/bias*
_output_shapes	
:?*
dtype0
?
8ActorDistributionNetwork/EncodingNetwork/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*I
shared_name:8ActorDistributionNetwork/EncodingNetwork/dense_11/kernel
?
LActorDistributionNetwork/EncodingNetwork/dense_11/kernel/Read/ReadVariableOpReadVariableOp8ActorDistributionNetwork/EncodingNetwork/dense_11/kernel* 
_output_shapes
:
??*
dtype0
?
6ActorDistributionNetwork/EncodingNetwork/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*G
shared_name86ActorDistributionNetwork/EncodingNetwork/dense_11/bias
?
JActorDistributionNetwork/EncodingNetwork/dense_11/bias/Read/ReadVariableOpReadVariableOp6ActorDistributionNetwork/EncodingNetwork/dense_11/bias*
_output_shapes	
:?*
dtype0
?
BActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/bias
?
VActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/bias/Read/ReadVariableOpReadVariableOpBActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/bias*
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
?
,ValueNetwork/EncodingNetwork/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*=
shared_name.,ValueNetwork/EncodingNetwork/dense_12/kernel
?
@ValueNetwork/EncodingNetwork/dense_12/kernel/Read/ReadVariableOpReadVariableOp,ValueNetwork/EncodingNetwork/dense_12/kernel*
_output_shapes
:	?*
dtype0
?
*ValueNetwork/EncodingNetwork/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*ValueNetwork/EncodingNetwork/dense_12/bias
?
>ValueNetwork/EncodingNetwork/dense_12/bias/Read/ReadVariableOpReadVariableOp*ValueNetwork/EncodingNetwork/dense_12/bias*
_output_shapes	
:?*
dtype0
?
,ValueNetwork/EncodingNetwork/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,ValueNetwork/EncodingNetwork/dense_13/kernel
?
@ValueNetwork/EncodingNetwork/dense_13/kernel/Read/ReadVariableOpReadVariableOp,ValueNetwork/EncodingNetwork/dense_13/kernel* 
_output_shapes
:
??*
dtype0
?
*ValueNetwork/EncodingNetwork/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*;
shared_name,*ValueNetwork/EncodingNetwork/dense_13/bias
?
>ValueNetwork/EncodingNetwork/dense_13/bias/Read/ReadVariableOpReadVariableOp*ValueNetwork/EncodingNetwork/dense_13/bias*
_output_shapes	
:?*
dtype0
?
ValueNetwork/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*-
shared_nameValueNetwork/dense_14/kernel
?
0ValueNetwork/dense_14/kernel/Read/ReadVariableOpReadVariableOpValueNetwork/dense_14/kernel*
_output_shapes
:	?*
dtype0
?
ValueNetwork/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameValueNetwork/dense_14/bias
?
.ValueNetwork/dense_14/bias/Read/ReadVariableOpReadVariableOpValueNetwork/dense_14/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
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
^
0
1
2
	3

4
5
6
7
8
9
10
11
12

0
 
zx
VARIABLE_VALUE8ActorDistributionNetwork/EncodingNetwork/dense_10/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE6ActorDistributionNetwork/EncodingNetwork/dense_10/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE8ActorDistributionNetwork/EncodingNetwork/dense_11/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE6ActorDistributionNetwork/EncodingNetwork/dense_11/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/bias,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUENActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUELActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE,ValueNetwork/EncodingNetwork/dense_12/kernel,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE*ValueNetwork/EncodingNetwork/dense_12/bias,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE,ValueNetwork/EncodingNetwork/dense_13/kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE*ValueNetwork/EncodingNetwork/dense_13/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEValueNetwork/dense_14/kernel-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEValueNetwork/dense_14/bias-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE

ref
1
(
_actor_network
_value_network
z
_encoder
_projection_networks
regularization_losses
trainable_variables
	variables
	keras_api
|
_encoder
_postprocessing_layers
regularization_losses
 trainable_variables
!	variables
"	keras_api
n
#_postprocessing_layers
$regularization_losses
%trainable_variables
&	variables
'	keras_api
z
(_means_projection_layer
	)_bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
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
regularization_losses

.layers
trainable_variables
/layer_metrics
0layer_regularization_losses
	variables
1non_trainable_variables
2metrics
n
3_postprocessing_layers
4regularization_losses
5trainable_variables
6	variables
7	keras_api
h

kernel
bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
?
regularization_losses

<layers
 trainable_variables
=layer_metrics
>layer_regularization_losses
!	variables
?non_trainable_variables
@metrics

A0
B1
C2
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
$regularization_losses

Dlayers
%trainable_variables
Elayer_metrics
Flayer_regularization_losses
&	variables
Gnon_trainable_variables
Hmetrics
h

kernel
bias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
\

bias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
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
*regularization_losses

Qlayers
+trainable_variables
Rlayer_metrics
Slayer_regularization_losses
,	variables
Tnon_trainable_variables
Umetrics

0
1
 
 
 
 

V0
W1
X2
 

0
1
2
3

0
1
2
3
?
4regularization_losses

Ylayers
5trainable_variables
Zlayer_metrics
[layer_regularization_losses
6	variables
\non_trainable_variables
]metrics
 

0
1

0
1
?
8regularization_losses

^layers
9trainable_variables
_layer_metrics
`layer_regularization_losses
:	variables
anon_trainable_variables
bmetrics

0
1
 
 
 
 
R
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
h

kernel
bias
gregularization_losses
htrainable_variables
i	variables
j	keras_api
h

kernel
	bias
kregularization_losses
ltrainable_variables
m	variables
n	keras_api

A0
B1
C2
 
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
Iregularization_losses

olayers
Jtrainable_variables
player_metrics
qlayer_regularization_losses
K	variables
rnon_trainable_variables
smetrics
 


0


0
?
Mregularization_losses

tlayers
Ntrainable_variables
ulayer_metrics
vlayer_regularization_losses
O	variables
wnon_trainable_variables
xmetrics

(0
)1
 
 
 
 
R
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
i

kernel
bias
}regularization_losses
~trainable_variables
	variables
?	keras_api
l

kernel
bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api

V0
W1
X2
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
?
cregularization_losses
?layers
dtrainable_variables
?layer_metrics
 ?layer_regularization_losses
e	variables
?non_trainable_variables
?metrics
 

0
1

0
1
?
gregularization_losses
?layers
htrainable_variables
?layer_metrics
 ?layer_regularization_losses
i	variables
?non_trainable_variables
?metrics
 

0
	1

0
	1
?
kregularization_losses
?layers
ltrainable_variables
?layer_metrics
 ?layer_regularization_losses
m	variables
?non_trainable_variables
?metrics
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
?
yregularization_losses
?layers
ztrainable_variables
?layer_metrics
 ?layer_regularization_losses
{	variables
?non_trainable_variables
?metrics
 

0
1

0
1
?
}regularization_losses
?layers
~trainable_variables
?layer_metrics
 ?layer_regularization_losses
	variables
?non_trainable_variables
?metrics
 

0
1

0
1
?
?regularization_losses
?layers
?trainable_variables
?layer_metrics
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
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
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type8ActorDistributionNetwork/EncodingNetwork/dense_10/kernel6ActorDistributionNetwork/EncodingNetwork/dense_10/bias8ActorDistributionNetwork/EncodingNetwork/dense_11/kernel6ActorDistributionNetwork/EncodingNetwork/dense_11/biasNActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernelLActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/biasBActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/bias*
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
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_10949163
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
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_10949175
?
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_10949197
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
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_10949190
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpLActorDistributionNetwork/EncodingNetwork/dense_10/kernel/Read/ReadVariableOpJActorDistributionNetwork/EncodingNetwork/dense_10/bias/Read/ReadVariableOpLActorDistributionNetwork/EncodingNetwork/dense_11/kernel/Read/ReadVariableOpJActorDistributionNetwork/EncodingNetwork/dense_11/bias/Read/ReadVariableOpVActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/bias/Read/ReadVariableOpbActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel/Read/ReadVariableOp`ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias/Read/ReadVariableOp@ValueNetwork/EncodingNetwork/dense_12/kernel/Read/ReadVariableOp>ValueNetwork/EncodingNetwork/dense_12/bias/Read/ReadVariableOp@ValueNetwork/EncodingNetwork/dense_13/kernel/Read/ReadVariableOp>ValueNetwork/EncodingNetwork/dense_13/bias/Read/ReadVariableOp0ValueNetwork/dense_14/kernel/Read/ReadVariableOp.ValueNetwork/dense_14/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_10949569
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable8ActorDistributionNetwork/EncodingNetwork/dense_10/kernel6ActorDistributionNetwork/EncodingNetwork/dense_10/bias8ActorDistributionNetwork/EncodingNetwork/dense_11/kernel6ActorDistributionNetwork/EncodingNetwork/dense_11/biasBActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/biasNActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernelLActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias,ValueNetwork/EncodingNetwork/dense_12/kernel*ValueNetwork/EncodingNetwork/dense_12/bias,ValueNetwork/EncodingNetwork/dense_13/kernel*ValueNetwork/EncodingNetwork/dense_13/biasValueNetwork/dense_14/kernelValueNetwork/dense_14/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_10949621??

?
l
,__inference_function_with_signature_10949182
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
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference_<lambda>_109488132
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
?.
?	
!__inference__traced_save_10949569
file_prefix'
#savev2_variable_read_readvariableop	W
Ssavev2_actordistributionnetwork_encodingnetwork_dense_10_kernel_read_readvariableopU
Qsavev2_actordistributionnetwork_encodingnetwork_dense_10_bias_read_readvariableopW
Ssavev2_actordistributionnetwork_encodingnetwork_dense_11_kernel_read_readvariableopU
Qsavev2_actordistributionnetwork_encodingnetwork_dense_11_bias_read_readvariableopa
]savev2_actordistributionnetwork_normalprojectionnetwork_bias_layer_2_bias_read_readvariableopm
isavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernel_read_readvariableopk
gsavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_bias_read_readvariableopK
Gsavev2_valuenetwork_encodingnetwork_dense_12_kernel_read_readvariableopI
Esavev2_valuenetwork_encodingnetwork_dense_12_bias_read_readvariableopK
Gsavev2_valuenetwork_encodingnetwork_dense_13_kernel_read_readvariableopI
Esavev2_valuenetwork_encodingnetwork_dense_13_bias_read_readvariableop;
7savev2_valuenetwork_dense_14_kernel_read_readvariableop9
5savev2_valuenetwork_dense_14_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopSsavev2_actordistributionnetwork_encodingnetwork_dense_10_kernel_read_readvariableopQsavev2_actordistributionnetwork_encodingnetwork_dense_10_bias_read_readvariableopSsavev2_actordistributionnetwork_encodingnetwork_dense_11_kernel_read_readvariableopQsavev2_actordistributionnetwork_encodingnetwork_dense_11_bias_read_readvariableop]savev2_actordistributionnetwork_normalprojectionnetwork_bias_layer_2_bias_read_readvariableopisavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernel_read_readvariableopgsavev2_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_bias_read_readvariableopGsavev2_valuenetwork_encodingnetwork_dense_12_kernel_read_readvariableopEsavev2_valuenetwork_encodingnetwork_dense_12_bias_read_readvariableopGsavev2_valuenetwork_encodingnetwork_dense_13_kernel_read_readvariableopEsavev2_valuenetwork_encodingnetwork_dense_13_bias_read_readvariableop7savev2_valuenetwork_dense_14_kernel_read_readvariableop5savev2_valuenetwork_dense_14_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
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

identity_1Identity_1:output:0*?
_input_shapesz
x: : :	?:?:
??:?::	?::	?:?:
??:?:	?:: 2(
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
::%	!

_output_shapes
:	?:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
ܳ
?
*__inference_polymorphic_action_fn_10949122
	time_step
time_step_1
time_step_2
time_step_3c
Pactordistributionnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource:	?`
Qactordistributionnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource:	?d
Pactordistributionnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource:
??`
Qactordistributionnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource:	?y
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:	?u
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:k
]actordistributionnetwork_normalprojectionnetwork_bias_layer_2_biasadd_readvariableop_resource:
identity??HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2:
8ActorDistributionNetwork/EncodingNetwork/flatten_4/Const?
:ActorDistributionNetwork/EncodingNetwork/flatten_4/ReshapeReshapetime_step_3AActorDistributionNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2<
:ActorDistributionNetwork/EncodingNetwork/flatten_4/Reshape?
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_10/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/flatten_4/Reshape:output:0OActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_10/MatMul?
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpQactordistributionnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02J
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAddBiasAddBActorDistributionNetwork/EncodingNetwork/dense_10/MatMul:product:0PActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd?
6ActorDistributionNetwork/EncodingNetwork/dense_10/ReluReluBActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense_10/Relu?
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_11/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/dense_10/Relu:activations:0OActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_11/MatMul?
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOpQactordistributionnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02J
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAddBiasAddBActorDistributionNetwork/EncodingNetwork/dense_11/MatMul:product:0PActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd?
6ActorDistributionNetwork/EncodingNetwork/dense_11/ReluReluBActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense_11/Relu?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02_
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/dense_11/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
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
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOpReadVariableOp]actordistributionnetwork_normalprojectionnetwork_bias_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02V
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp?
EActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAddBiasAdd?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0\ActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2G
EActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape?
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeNActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
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
IdentityIdentityclip_by_value:z:0I^ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpI^ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpU^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:?????????: : : : : : : 2?
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpHActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2?
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpHActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp2?
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOpTActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp2?
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
>
,__inference_function_with_signature_10949170

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
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_get_initial_state_109491692
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
?
d
__inference_<lambda>_10948813!
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
ҳ
?
*__inference_polymorphic_action_fn_10949303
	step_type

reward
discount
observationc
Pactordistributionnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource:	?`
Qactordistributionnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource:	?d
Pactordistributionnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource:
??`
Qactordistributionnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource:	?y
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:	?u
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:k
]actordistributionnetwork_normalprojectionnetwork_bias_layer_2_biasadd_readvariableop_resource:
identity??HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2:
8ActorDistributionNetwork/EncodingNetwork/flatten_4/Const?
:ActorDistributionNetwork/EncodingNetwork/flatten_4/ReshapeReshapeobservationAActorDistributionNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2<
:ActorDistributionNetwork/EncodingNetwork/flatten_4/Reshape?
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_10/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/flatten_4/Reshape:output:0OActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_10/MatMul?
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpQactordistributionnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02J
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAddBiasAddBActorDistributionNetwork/EncodingNetwork/dense_10/MatMul:product:0PActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd?
6ActorDistributionNetwork/EncodingNetwork/dense_10/ReluReluBActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense_10/Relu?
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_11/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/dense_10/Relu:activations:0OActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_11/MatMul?
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOpQactordistributionnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02J
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAddBiasAddBActorDistributionNetwork/EncodingNetwork/dense_11/MatMul:product:0PActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd?
6ActorDistributionNetwork/EncodingNetwork/dense_11/ReluReluBActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense_11/Relu?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02_
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/dense_11/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
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
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOpReadVariableOp]actordistributionnetwork_normalprojectionnetwork_bias_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02V
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp?
EActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAddBiasAdd?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0\ActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2G
EActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape?
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeNActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
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
IdentityIdentityclip_by_value:z:0I^ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpI^ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpU^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:?????????: : : : : : : 2?
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpHActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2?
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpHActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp2?
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOpTActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp2?
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
?
(
&__inference_signature_wrapper_10949197?
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
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_109491932
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
,__inference_function_with_signature_10949139
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
*0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_polymorphic_action_fn_109491222
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
8
&__inference_get_initial_state_10949499

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
?
&__inference_signature_wrapper_10949163
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
*0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_109491392
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
?
8
&__inference_signature_wrapper_10949175

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
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_109491702
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
?
8
&__inference_get_initial_state_10949169

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
^

__inference_<lambda>_10948816*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
*__inference_polymorphic_action_fn_10949408
time_step_step_type
time_step_reward
time_step_discount
time_step_observationc
Pactordistributionnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource:	?`
Qactordistributionnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource:	?d
Pactordistributionnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource:
??`
Qactordistributionnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource:	?y
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:	?u
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:k
]actordistributionnetwork_normalprojectionnetwork_bias_layer_2_biasadd_readvariableop_resource:
identity??HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2:
8ActorDistributionNetwork/EncodingNetwork/flatten_4/Const?
:ActorDistributionNetwork/EncodingNetwork/flatten_4/ReshapeReshapetime_step_observationAActorDistributionNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2<
:ActorDistributionNetwork/EncodingNetwork/flatten_4/Reshape?
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_10/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/flatten_4/Reshape:output:0OActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_10/MatMul?
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpQactordistributionnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02J
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAddBiasAddBActorDistributionNetwork/EncodingNetwork/dense_10/MatMul:product:0PActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd?
6ActorDistributionNetwork/EncodingNetwork/dense_10/ReluReluBActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense_10/Relu?
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_11/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/dense_10/Relu:activations:0OActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_11/MatMul?
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOpQactordistributionnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02J
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAddBiasAddBActorDistributionNetwork/EncodingNetwork/dense_11/MatMul:product:0PActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd?
6ActorDistributionNetwork/EncodingNetwork/dense_11/ReluReluBActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense_11/Relu?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02_
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/dense_11/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
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
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOpReadVariableOp]actordistributionnetwork_normalprojectionnetwork_bias_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02V
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp?
EActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAddBiasAdd?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0\ActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2G
EActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape?
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeNActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
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
IdentityIdentityclip_by_value:z:0I^ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpI^ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpU^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????:?????????:?????????: : : : : : : 2?
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpHActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2?
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpHActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp2?
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOpTActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp2?
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
?D
?
$__inference__traced_restore_10949621
file_prefix#
assignvariableop_variable:	 ^
Kassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_10_kernel:	?X
Iassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_10_bias:	?_
Kassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_11_kernel:
??X
Iassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_11_bias:	?c
Uassignvariableop_5_actordistributionnetwork_normalprojectionnetwork_bias_layer_2_bias:t
aassignvariableop_6_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_kernel:	?m
_assignvariableop_7_actordistributionnetwork_normalprojectionnetwork_means_projection_layer_bias:R
?assignvariableop_8_valuenetwork_encodingnetwork_dense_12_kernel:	?L
=assignvariableop_9_valuenetwork_encodingnetwork_dense_12_bias:	?T
@assignvariableop_10_valuenetwork_encodingnetwork_dense_13_kernel:
??M
>assignvariableop_11_valuenetwork_encodingnetwork_dense_13_bias:	?C
0assignvariableop_12_valuenetwork_dense_14_kernel:	?<
.assignvariableop_13_valuenetwork_dense_14_bias:
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	2
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
AssignVariableOp_1AssignVariableOpKassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_10_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpIassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_10_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpKassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_11_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpIassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_11_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpUassignvariableop_5_actordistributionnetwork_normalprojectionnetwork_bias_layer_2_biasIdentity_5:output:0"/device:CPU:0*
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
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp?assignvariableop_8_valuenetwork_encodingnetwork_dense_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp=assignvariableop_9_valuenetwork_encodingnetwork_dense_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp@assignvariableop_10_valuenetwork_encodingnetwork_dense_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp>assignvariableop_11_valuenetwork_encodingnetwork_dense_13_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp0assignvariableop_12_valuenetwork_dense_14_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp.assignvariableop_13_valuenetwork_dense_14_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14?
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
f
&__inference_signature_wrapper_10949190
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
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_109491822
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
?
.
,__inference_function_with_signature_10949193?
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
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference_<lambda>_109488162
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
0__inference_polymorphic_distribution_fn_10949496
	step_type

reward
discount
observationc
Pactordistributionnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource:	?`
Qactordistributionnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource:	?d
Pactordistributionnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource:
??`
Qactordistributionnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource:	?y
factordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:	?u
gactordistributionnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:k
]actordistributionnetwork_normalprojectionnetwork_bias_layer_2_biasadd_readvariableop_resource:
identity??HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp?^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp?]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2:
8ActorDistributionNetwork/EncodingNetwork/flatten_4/Const?
:ActorDistributionNetwork/EncodingNetwork/flatten_4/ReshapeReshapeobservationAActorDistributionNetwork/EncodingNetwork/flatten_4/Const:output:0*
T0*'
_output_shapes
:?????????2<
:ActorDistributionNetwork/EncodingNetwork/flatten_4/Reshape?
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_10_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_10/MatMulMatMulCActorDistributionNetwork/EncodingNetwork/flatten_4/Reshape:output:0OActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_10/MatMul?
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpReadVariableOpQactordistributionnetwork_encodingnetwork_dense_10_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02J
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAddBiasAddBActorDistributionNetwork/EncodingNetwork/dense_10/MatMul:product:0PActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd?
6ActorDistributionNetwork/EncodingNetwork/dense_10/ReluReluBActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd:output:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense_10/Relu?
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_11_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp?
8ActorDistributionNetwork/EncodingNetwork/dense_11/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/dense_10/Relu:activations:0OActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2:
8ActorDistributionNetwork/EncodingNetwork/dense_11/MatMul?
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpReadVariableOpQactordistributionnetwork_encodingnetwork_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02J
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp?
9ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAddBiasAddBActorDistributionNetwork/EncodingNetwork/dense_11/MatMul:product:0PActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2;
9ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd?
6ActorDistributionNetwork/EncodingNetwork/dense_11/ReluReluBActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????28
6ActorDistributionNetwork/EncodingNetwork/dense_11/Relu?
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpfactordistributionnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02_
]ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp?
NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulDActorDistributionNetwork/EncodingNetwork/dense_11/Relu:activations:0eActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
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
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOpReadVariableOp]actordistributionnetwork_normalprojectionnetwork_bias_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02V
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp?
EActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAddBiasAdd?ActorDistributionNetwork/NormalProjectionNetwork/zeros_like:y:0\ActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2G
EActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd?
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2B
@ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape?
:ActorDistributionNetwork/NormalProjectionNetwork/Reshape_1ReshapeNActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd:output:0IActorDistributionNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
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
IdentityIdentityTActorDistributionNetwork_NormalProjectionNetwork_MultivariateNormalDiag/mode/add:z:0I^ActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpI^ActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpU^ActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp_^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp^^ActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
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
HActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOpHActorDistributionNetwork/EncodingNetwork/dense_10/BiasAdd/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_10/MatMul/ReadVariableOp2?
HActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOpHActorDistributionNetwork/EncodingNetwork/dense_11/BiasAdd/ReadVariableOp2?
GActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_11/MatMul/ReadVariableOp2?
TActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOpTActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/BiasAdd/ReadVariableOp2?
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
_user_specified_nameobservation"?L
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
?action
?distribution
?get_initial_state
?get_metadata
?get_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper

0
1
2
	3

4
5
6
7
8
9
10
11
12"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
d
?action
?get_initial_state
?get_train_step
?get_metadata"
signature_map
K:I	?28ActorDistributionNetwork/EncodingNetwork/dense_10/kernel
E:C?26ActorDistributionNetwork/EncodingNetwork/dense_10/bias
L:J
??28ActorDistributionNetwork/EncodingNetwork/dense_11/kernel
E:C?26ActorDistributionNetwork/EncodingNetwork/dense_11/bias
P:N2BActorDistributionNetwork/NormalProjectionNetwork/bias_layer_2/bias
a:_	?2NActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/kernel
Z:X2LActorDistributionNetwork/NormalProjectionNetwork/means_projection_layer/bias
?:=	?2,ValueNetwork/EncodingNetwork/dense_12/kernel
9:7?2*ValueNetwork/EncodingNetwork/dense_12/bias
@:>
??2,ValueNetwork/EncodingNetwork/dense_13/kernel
9:7?2*ValueNetwork/EncodingNetwork/dense_13/bias
/:-	?2ValueNetwork/dense_14/kernel
(:&2ValueNetwork/dense_14/bias
1
ref
1"
trackable_tuple_wrapper
F
_actor_network
_value_network"
_generic_user_object
?
_encoder
_projection_networks
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "ActorDistributionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ActorDistributionNetwork", "config": {"layer was saved without config": true}}
?
_encoder
_postprocessing_layers
regularization_losses
 trainable_variables
!	variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "ValueNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ValueNetwork", "config": {"layer was saved without config": true}}
?
#_postprocessing_layers
$regularization_losses
%trainable_variables
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EncodingNetwork", "config": {"layer was saved without config": true}}
?
(_means_projection_layer
	)_bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
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
regularization_losses

.layers
trainable_variables
/layer_metrics
0layer_regularization_losses
	variables
1non_trainable_variables
2metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
3_postprocessing_layers
4regularization_losses
5trainable_variables
6	variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EncodingNetwork", "config": {"layer was saved without config": true}}
?

kernel
bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
regularization_losses

<layers
 trainable_variables
=layer_metrics
>layer_regularization_losses
!	variables
?non_trainable_variables
@metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5
A0
B1
C2"
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
$regularization_losses

Dlayers
%trainable_variables
Elayer_metrics
Flayer_regularization_losses
&	variables
Gnon_trainable_variables
Hmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

kernel
bias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "means_projection_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "means_projection_layer", "trainable": true, "dtype": "float32", "units": 30, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.1, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
?

bias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "bias_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BiasLayer", "config": {"name": "bias_layer_2", "trainable": true, "dtype": "float32", "bias_initializer": {"class_name": "Constant", "config": {"value": -0.8697231582271624}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 30]}}
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
*regularization_losses

Qlayers
+trainable_variables
Rlayer_metrics
Slayer_regularization_losses
,	variables
Tnon_trainable_variables
Umetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
V0
W1
X2"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
4regularization_losses

Ylayers
5trainable_variables
Zlayer_metrics
[layer_regularization_losses
6	variables
\non_trainable_variables
]metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
8regularization_losses

^layers
9trainable_variables
_layer_metrics
`layer_regularization_losses
:	variables
anon_trainable_variables
bmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
gregularization_losses
htrainable_variables
i	variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 30]}}
?

kernel
	bias
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
5
A0
B1
C2"
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Iregularization_losses

olayers
Jtrainable_variables
player_metrics
qlayer_regularization_losses
K	variables
rnon_trainable_variables
smetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
Mregularization_losses

tlayers
Ntrainable_variables
ulayer_metrics
vlayer_regularization_losses
O	variables
wnon_trainable_variables
xmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
yregularization_losses
ztrainable_variables
{	variables
|	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
}regularization_losses
~trainable_variables
	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 30]}}
?

kernel
bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256]}}
5
V0
W1
X2"
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
trackable_list_wrapper
 "
trackable_list_wrapper
?
cregularization_losses
?layers
dtrainable_variables
?layer_metrics
 ?layer_regularization_losses
e	variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
gregularization_losses
?layers
htrainable_variables
?layer_metrics
 ?layer_regularization_losses
i	variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
kregularization_losses
?layers
ltrainable_variables
?layer_metrics
 ?layer_regularization_losses
m	variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
trackable_list_wrapper
 "
trackable_list_wrapper
?
yregularization_losses
?layers
ztrainable_variables
?layer_metrics
 ?layer_regularization_losses
{	variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
}regularization_losses
?layers
~trainable_variables
?layer_metrics
 ?layer_regularization_losses
	variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?regularization_losses
?layers
?trainable_variables
?layer_metrics
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
?2?
*__inference_polymorphic_action_fn_10949303
*__inference_polymorphic_action_fn_10949408?
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
0__inference_polymorphic_distribution_fn_10949496?
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
&__inference_get_initial_state_10949499?
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
__inference_<lambda>_10948816"?
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
__inference_<lambda>_10948813"?
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
&__inference_signature_wrapper_10949163
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
&__inference_signature_wrapper_10949175
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
&__inference_signature_wrapper_10949190"?
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
&__inference_signature_wrapper_10949197"?
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
 <
__inference_<lambda>_10948813?

? 
? "? 	5
__inference_<lambda>_10948816?

? 
? "? S
&__inference_get_initial_state_10949499)"?
?
?

batch_size 
? "? ?
*__inference_polymorphic_action_fn_10949303?	
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
*__inference_polymorphic_action_fn_10949408?	
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
0__inference_polymorphic_distribution_fn_10949496?	
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
&__inference_signature_wrapper_10949163?	
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
action?????????a
&__inference_signature_wrapper_1094917570?-
? 
&?#
!

batch_size?

batch_size "? Z
&__inference_signature_wrapper_109491900?

? 
? "?

int64?
int64 	>
&__inference_signature_wrapper_10949197?

? 
? "? 