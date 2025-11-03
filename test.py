
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

from motion.obstacle import ObstacleMap
from motion.agent import Agent, Env
from visualize import Visualizer


if __name__ == '__main__':
    omap = ObstacleMap()
    omap.set_circle('obstacle1', 100, 100, 20)
    
    agent0 = Agent('a0', [0, 0, 0, 0], [200, 200, 0, 0], steps=8, radius=5, omap=omap)
    agent1 = Agent('a1', [0, 200, 0, 0], [200, 0, 0, 0], steps=8, radius=5, omap=omap)
    agent2 = Agent('a2', [200, 200, 0, 0], [0, 0, 0, 0], steps=8, radius=5, omap=omap)
    agent3 = Agent('a3', [200, 0, 0, 0], [0, 200, 0, 0], steps=8, radius=5, omap=omap)
    agent4 = Agent('a4', [100, 0, 0, 0], [100, 200, 0, 0], steps=8, radius=5, omap=omap)
    agent5 = Agent('a5', [0, 100, 0, 0], [200, 100, 0, 0], steps=8, radius=5, omap=omap)
    agent6 = Agent('a6', [200, 100, 0, 0], [0, 100, 0, 0], steps=8, radius=5, omap=omap)
    agent7 = Agent('a7', [100, 200, 0, 0], [100, 0, 0, 0], steps=8, radius=5, omap=omap)
    
    # 환경 설정
    env = Env()
    env.add_agent(agent0)
    env.add_agent(agent1)
    env.add_agent(agent2)
    env.add_agent(agent3)
    env.add_agent(agent4)
    env.add_agent(agent5)
    env.add_agent(agent6)
    env.add_agent(agent7)
    
    agents = [agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7]
    
    visualizer = Visualizer(omap, agents)
    
    print("종료: Ctrl+C 또는 창 닫기")
    
    step_count = 0
    
    try:
        while True:
            # GBP 기반 경로 계획
            env.step_plan(iters=5)
            
            # 시각화 업데이트
            visualizer.update_visualization()
            
            # 에이전트 이동
            env.step_move()
            
            # 진행 상황 출력
            if step_count % 10 == 0:
                print(f"Step {step_count}")
                all_reached_in_step = True
                for agent in agents:
                    state = agent.get_state()
                    if state[0] is not None:
                        x, y, vx, vy = state[0]
                        target = agent.get_target()
                        dist_to_goal = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                        print(f"  {agent.name}: pos=({x:.1f}, {y:.1f}), dist_to_goal={dist_to_goal:.1f}")
                        if dist_to_goal > 10:
                            all_reached_in_step = False
                    else:
                        all_reached_in_step = False 
                
                if all_reached_in_step:
                    print("\n(출력) 모든 에이전트가 목표 근처 도달 시작")
    
            # time.sleep(0.001) 
            
            step_count += 1
            
            # 모든 에이전트가 목표에 도달했는지 확인
            all_reached = True
            for agent in agents:
                state = agent.get_state()
                if state[0] is not None: 
                    x, y, _, _ = state[0]
                    target = agent.get_target()
                    dist = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                    if dist > 10:  
                        all_reached = False
                        break
                else: 
                    pass 
            
            if all_reached:
                print("\n모든 에이전트가 목표에 도달했습니다!")
                print("\n10초후에 시뮬레이션을 종료합니다.")
                plt.pause(10) # 10초간 마지막 모습 보여주기
                break
                
    except KeyboardInterrupt:
        print("\n시뮬레이션 중단됨")
    
    finally:
        print("시뮬레이션 종료")
        visualizer.close()