# AgileX LIMO ROS 1 Robot Pack

This Pack binds the `limo` e-URDF Body to the independently versioned
`ros-claw/limo-ros-mcp` adapter at commit
`fd9275c9de22f6158a38edc4b299e6657bce38bb` (MCP 0.6.5).

The first REAL capability is `limo.set_initial_pose`. The Agent submits a
validated map-frame estimate to `rosclawd`; the daemon validates an exact
operator permit and starts a fixed-operation ROS Melodic worker. The Agent and
MCP server do not import rospy or publish `/initialpose`.

H1 means only that the signed contract and executor tests pass. H4 requires a
real LIMO, an AMCL subscriber, post-dispatch `/amcl_pose`, a `map -> odom`
transform, a canonical TASK_VERIFIED receipt, and independent review.

Revision 0.1.2 waits for a converged AMCL sample within the bounded
verification window, so an immediate transient `/amcl_pose` sample does not
produce a false-negative receipt.

Revision 0.1.3 adds in-context MCP operator confirmation for REAL initial-pose
requests and allows five seconds for read-only ROS CLI startup on LIMO ARM
hosts. Clients without MCP form elicitation now fail immediately instead of
waiting for a confirmation response they cannot render. Permit material remains
internal to the trusted ROSClaw host boundary.

Revision 0.1.4 matches the Dabai U3 streams launched by
`astra_camera/dabai_u3.launch`: `/camera/color/image_raw`,
`/camera/depth/image_raw`, and `/camera/depth/points`. The MCP returns bounded
metadata summaries and never exposes the raw image or point-cloud arrays.
ROS CLI array placeholders are decoded into their true bounded byte and field
counts, rather than reporting the placeholder string length.
