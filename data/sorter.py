import time

# moski_poklici = open("poklici-moski.txt", "r")
#
# m_poklici = moski_poklici.readlines()
#
# zenski_poklici = open("poklici-zenski.txt")
# z_poklici = zenski_poklici.readlines()
#
#
# moski_sorted = sorted(m_poklici)
# zenski_sorted = sorted(z_poklici)

moski_poklici = open("sorted_moski.txt", "r")

zenski_poklici = open("sorted_zenski.txt")

moski_sorted = moski_poklici.readlines()
zenski_sorted = zenski_poklici.readlines()

for i in range(len(moski_sorted)):
    print(moski_sorted[i], zenski_sorted[i])


# sz = open("sorted_zenski.txt", "w")
# sm = open("sorted_moski.txt", "w")
#
# sz.writelines(zenski_sorted)
# sm.writelines(moski_sorted)

# sz.close()
# sm.close()
zenski_poklici.close()
moski_poklici.close()










