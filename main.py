"""
Main file that executes the visual conversation part.
"""

from conversation import dodeca_statemachine

statemachine = dodeca_statemachine.DodecaStateMachine()

while True:
    statemachine.run()
