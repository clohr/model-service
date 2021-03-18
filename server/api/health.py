from server.db import Database


def get_health() -> int:
    db = Database.from_server()
    m = db.get_one()
    if m == 1:
        return 200
    return 500
