Put your exported TorchScript policy here:

  policy/bpx_dwaq_v2.pt

The UI default path is this file. For the current IsaacLab WAQ export, the
default policy is:

  /home/ubuntu/bpx_simreal_v6/bpx_simreal_v6_crouchfix_xwk/policy/bpx_dwaq_v2.pt

Expected exported policy interface:
  input  shape: [1, 675]  = 15 * 45 DWAQ obs history
  output shape: [1, 12]   = 12 leg joint actions in rl_joint_order_12

Alternative wrapper:
  policy/bpx_dwaq_v2_current_plus_history.pt
  input shape: [1, 720] = 45 current obs + 15 * 45 history
