import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np
import time

from motion.obstacle import ObstacleMap
from motion.agent import Agent, Env


class PyBulletVisualizer:
    def __init__(self, omap: ObstacleMap, agents: list):
        # PyBullet 초기화
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        # 카메라 설정
        p.resetDebugVisualizerCamera(
            cameraDistance=200,
            cameraYaw=0,
            cameraPitch=-89,  # 위에서 내려다보는 시점
            cameraTargetPosition=[100, 100, 0]
        )
        
        # 바닥 생성
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.9, 0.9, 0.9, 1])
        
        self.omap = omap
        self.agents = agents
        
        # 장애물 시각화
        self.obstacle_ids = []
        self._create_obstacles()
        
        # 에이전트 시각화 객체
        self.agent_bodies = {}
        cmap = plt.cm.get_cmap('tab10', len(agents))
        self.agent_colors = {agent: list(cmap(i)[:4]) for i, agent in enumerate(agents)}
        self._create_agents()
        
        # 궤적 선 저장
        self.trajectory_lines = {agent: [] for agent in agents}
        
    def _create_obstacles(self):
        """장애물을 PyBullet 객체로 생성"""
        for name, obj in self.omap.objects.items():
            if obj['type'] == 'circle':
                # 원형 장애물을 실린더로 표현
                collision_shape = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=obj['radius'],
                    height=20
                )
                visual_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=obj['radius'],
                    length=20,
                    rgbaColor=[0.9, 0.2, 0.2, 0.8]
                )
                
                obstacle_id = p.createMultiBody(
                    baseMass=0,  # 고정 장애물
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[obj['centerx'], obj['centery'], 10]
                )
                self.obstacle_ids.append(obstacle_id)
    
    def _create_agents(self):
        """에이전트를 PyBullet 객체로 생성"""
        for agent in self.agents:
            # 구체로 에이전트 표현
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=agent.r
            )
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=agent.r,
                rgbaColor=self.agent_colors[agent]
            )
            
            body_id = p.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[agent.x, agent.y, agent.r]
            )
            
            self.agent_bodies[agent] = body_id
            
            # 목표 지점 표시 (작은 반투명 구체)
            target = agent.get_target()
            target_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=agent.r * 0.5,
                rgbaColor=self.agent_colors[agent][:3] + [0.3]
            )
            target_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=target_visual,
                basePosition=[target[0, 0], target[1, 0], agent.r * 0.5]
            )
    
    def update_visualization(self):
        """에이전트 위치 및 궤적 업데이트"""
        for agent in self.agents:
            body_id = self.agent_bodies[agent]
            
            # 에이전트 위치 업데이트
            state = agent.get_state()
            if state[0] is not None:
                x, y, vx, vy = state[0]
                p.resetBasePositionAndOrientation(
                    body_id,
                    [x, y, agent.r],
                    [0, 0, 0, 1]
                )
            
            # 이전 궤적 선 제거
            for line_id in self.trajectory_lines[agent]:
                p.removeUserDebugItem(line_id)
            self.trajectory_lines[agent].clear()
            
            # 계획된 궤적 그리기
            # color = self.agent_colors[agent][:3]
            # for i in range(len(state) - 1):
            #     if state[i] is None or state[i+1] is None:
            #         break
            #     x0, y0, _, _ = state[i]
            #     x1, y1, _, _ = state[i+1]
                
            #     line_id = p.addUserDebugLine(
            #         [x0, y0, agent.r],
            #         [x1, y1, agent.r],
            #         lineColorRGB=color,
            #         lineWidth=2,
            #         lifeTime=0
            #     )
            #     self.trajectory_lines[agent].append(line_id)
            
            # MPPI 궤적이 있다면 표시 (더 가는 선으로)
            color = self.agent_colors[agent][:3]
            mppi_traj = agent.get_trajectory()
            if mppi_traj is not None:
                for i in range(len(mppi_traj) - 1):
                    line_id = p.addUserDebugLine(
                        [mppi_traj[i][0, 0], mppi_traj[i][1, 0], agent.r + 2],
                        [mppi_traj[i+1][0, 0], mppi_traj[i+1][1, 0], agent.r + 2],
                        lineColorRGB=color,# 주황색
                        lineWidth=1,
                        lifeTime=0
                    )
                    self.trajectory_lines[agent].append(line_id)
    
    def close(self):
        p.disconnect()


if __name__ == '__main__':
    # 장애물 맵 생성
    omap = ObstacleMap()
    omap.set_circle('obstacle1', 145, 85, 20)
    
    # 에이전트 생성 (4개의 에이전트가 대각선으로 교차)
    agent0 = Agent('a0', [0, 0, 0, 0], [200, 200, 0, 0], steps=8, radius=5, omap=omap)
    agent1 = Agent('a1', [0, 200, 0, 0], [200, 0, 0, 0], steps=8, radius=5, omap=omap)
    agent2 = Agent('a2', [200, 200, 0, 0], [0, 0, 0, 0], steps=8, radius=5, omap=omap)
    agent3 = Agent('a3', [200, 0, 0, 0], [0, 200, 0, 0], steps=8, radius=5, omap=omap)
    agent4 = Agent('a4', [100, 0, 0, 0], [100, 200, 0, 0], steps=8, radius=5, omap=omap)
    agent5 = Agent('a5', [0, 100, 0, 0], [200, 100, 0, 0], steps=8, radius=5, omap=omap)
    
    # 환경 설정
    env = Env()
    env.add_agent(agent0)
    env.add_agent(agent1)
    env.add_agent(agent2)
    env.add_agent(agent3)
    env.add_agent(agent4)
    env.add_agent(agent5)
    
    agents = [agent0, agent1, agent2, agent3, agent4, agent5]
    
    # PyBullet 시각화 초기화
    visualizer = PyBulletVisualizer(omap, agents)
    
    print("시뮬레이션 시작...")
    print("카메라 조작: 마우스 드래그")
    print("종료: ESC 또는 창 닫기")
    
    step_count = 0
    max_steps = 200
    
    try:
        while step_count < max_steps:
            # GBP 기반 경로 계획
            env.step_plan(iters=5)
            
            # 시각화 업데이트
            visualizer.update_visualization()
            
            # 에이전트 이동
            env.step_move()
            
            # 진행 상황 출력
            if step_count % 10 == 0:
                print(f"Step {step_count}/{max_steps}")
                for agent in agents:
                    state = agent.get_state()
                    if state[0] is not None:
                        x, y, vx, vy = state[0]
                        target = agent.get_target()
                        dist_to_goal = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                        print(f"  {agent.name}: pos=({x:.1f}, {y:.1f}), dist_to_goal={dist_to_goal:.1f}")
            
            # 시뮬레이션 스텝
            p.stepSimulation()
            time.sleep(0.001)
            
            step_count += 1
            
            # 모든 에이전트가 목표에 도달했는지 확인
            all_reached = True
            for agent in agents:
                state = agent.get_state()
                if state[0] is not None:
                    x, y, _, _ = state[0]
                    target = agent.get_target()
                    dist = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                    if dist > 10:  # 목표까지 10 이내
                        all_reached = False
                        break
            
            if all_reached:
                print("\n모든 에이전트가 목표에 도달했습니다!")
                time.sleep(3)
                break
                
    except KeyboardInterrupt:
        print("\n시뮬레이션 중단됨")
    
    finally:
        print("시뮬레이션 종료")
        visualizer.close()