from typing import TypedDict

class User(TypedDict):
    employ_name: str
    employ_id: int
    
    
    
new_user: User= {

    "employ_name": "John Doe",
    "employ_id": 12345
}

print(new_user)