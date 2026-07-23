"""
Day 17-18: Simple hardcoded user/role system for RBAC demo purposes.

In a real enterprise system this would come from an identity provider
(SSO, LDAP, Azure AD, etc.), not a hardcoded dict - this is a deliberate,
documented simplification appropriate for a project-scale demo. The
important part this demonstrates isn't the auth mechanism itself, it's
that access control is enforced as a metadata filter INSIDE the retrieval
query - the part that's genuinely hard to bypass, unlike prompting the
LLM to "not mention" restricted content.
"""
from app.config import ROLE_CLEARANCE, SENSITIVITY_LEVELS

# username -> role. Add more test users here as needed.
USERS = {
    "intern_raj": "intern",
    "employee_amit": "employee",
    "manager_sana": "manager",
    "hr_priya": "hr",
    "admin_root": "admin",
}


def get_role(username: str) -> str:
    role = USERS.get(username)
    if role is None:
        raise ValueError(
            f"Unknown user '{username}'. Valid demo users: {list(USERS.keys())}"
        )
    return role


def get_allowed_sensitivities(role: str) -> list[str]:
    """
    Returns every sensitivity tag this role's clearance permits.
    A role can see anything at or below its own clearance level.
    """
    clearance = ROLE_CLEARANCE[role]
    return [tag for tag, level in SENSITIVITY_LEVELS.items() if level <= clearance]