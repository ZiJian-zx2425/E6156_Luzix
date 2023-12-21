users_db = {
    '110788464327696265201': 'patient',
    '104405107080836112407': 'doctor',
    '117740487543455173970': 'volunteer',
}

print(users_db.get('110788464327696265201', 'not logged in'))
print(users_db.get('104405107080836112407', 'not logged in'))
print(users_db.get('117740487543455173970', 'not logged in'))