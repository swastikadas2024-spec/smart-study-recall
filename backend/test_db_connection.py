
from sqlalchemy import text
from app.database import engine

if engine is None:
    print("ERROR: DATABASE_URL is not configured.")
else:
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1")).scalar_one()
            print("SUCCESS: Connected to PostgreSQL.")
            print("Test query result:", result)
    except Exception as exc:
        print("DATABASE CONNECTION FAILED")
        print("Error type:", type(exc).__name__)
        print("Details:", str(exc).splitlines()[0])
