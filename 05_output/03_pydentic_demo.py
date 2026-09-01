from pydantic import BaseModel,Field, EmailStr
from typing import Optional

class User(BaseModel):
    name: str  = "Jai"
    age: Optional[int] = None
    email: EmailStr 
    CGPA : float = Field(default=0.0, gt=0.0, lt=4.0)

user1 ={"email":"jai@example.com"}

user = User(**user1)


user_dic = dict(user)
user_json = user.model_dump_json()

print(user)
print(user_dic)
print(user_json)
 