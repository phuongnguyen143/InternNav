import habitat_sim

# Script tạo navmesh từ GLB
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = (
    "/home/lenguyen1/hoangpqn/InternNav/data/scene_data/mp3d/office/5_6_2026.glb"
)
backend_cfg.enable_physics = False

agent_cfg = habitat_sim.agent.AgentConfiguration()
sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

# Tạo và lưu navmesh
navmesh_settings = habitat_sim.NavMeshSettings()
navmesh_settings.set_defaults()
sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
sim.pathfinder.save_nav_mesh(
    "/home/lenguyen1/hoangpqn/InternNav/data/scene_data/mp3d/office/5_6_2026.navmesh"
)
print("Done! Navmesh saved.")
sim.close()
