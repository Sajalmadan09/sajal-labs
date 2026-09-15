"""Real dataset for exp9: common Indian first names labeled by conventional
gender, spanning multiple linguistic/religious naming traditions (North
Indian Hindu/Sanskrit-origin, South Indian Tamil/Telugu/Kannada/Malayalam,
Muslim, Sikh/Punjabi, Bengali, Gujarati, Marathi). Hand-compiled from general
knowledge of names in wide use, not scraped or fabricated — every name here
is one a person plausibly carries. Not an exhaustive or perfectly balanced
census (no claim of demographic representativeness), but a genuine dataset
for a genuine small-model teaching case study, per the brief's Model 4 and
its instruction not to fabricate data.

A name can appear in multiple regional lists (e.g. "Amit" is common in both
North Indian and Bengali/Marathi contexts) and a very small number of names
are used for both genders in practice (e.g. some Sikh names) — cross-list
duplicates are deduped case-insensitively below; the rare few names that
legitimately appear in both MALE and FEMALE lists are dropped entirely
(ambiguous label, not useful signal for a binary classifier) rather than
arbitrarily assigned.
"""
import csv
import pathlib

MALE_NAMES = """
Amit Rahul Rajesh Suresh Ramesh Vijay Sanjay Ajay Vikram Rohit Arjun Karan
Aditya Ankit Deepak Manoj Anil Sunil Pankaj Nitin Ashok Ravi Ravindra Mahesh
Naresh Dinesh Mukesh Rakesh Yogesh Hitesh Jitesh Umesh Paresh Mohan Sohan
Rohan Kishore Prakash Vinod Vishal Vivek Varun Nikhil Nishant Siddharth
Shubham Saurabh Gaurav Kunal Kartik Krishna Krishnan Govind Gopal Ganesh
Harish Harsh Hemant Anand Arvind Abhishek Aakash Akash Alok Amar Arun Ashish
Atul Bharat Bhanu Chetan Devendra Dhruv Girish Harendra Ishaan Jai Jatin
Jayesh Kailash Kamal Kapil Lalit Lokesh Madhav Manish Milind Mohit Narendra
Neeraj Om Parth Pradeep Pramod Pratap Puneet Rajat Rajeev Raju Rishabh Sachin
Sameer Sandeep Satish Shailesh Shantanu Shashank Shivam Shyam Sourabh Subhash
Surendra Swapnil Tarun Uday Uttam Vikas Vinay Vishnu Yash Yatin
Prasad Srinivas Srinivasan Venkatesh Venkataraman Raghavan Raghuram Ganapathy
Ganeshan Muthu Murugan Rajan Rajendran Ravichandran Sankar Shankar Balaji
Balasubramaniam Chandrasekhar Chandran Elango Gopalakrishnan Govindarajan
Ilango Jagadeesh Jayaraman Kannan Karthik Karthikeyan Kumaran Madhavan Mani
Manikandan Mohanraj Narayanan Natarajan Padmanabhan Palanisamy Prabhakar
Rajagopal Rajaraman Ramachandran Ramakrishnan Ramanathan Ramanujam
Ranganathan Sathish Selvam Senthil Shanmugam Sivakumar Sivaraman Subramaniam
Subramanian Sundaram Sundararajan Swaminathan Thangaraj Velu Venkataramanan
Vijayan Vishwanathan Chinna Anbu Arasu Bala Chezhian Gopi Karunanidhi Murali
Prabhu Raghu Ramu Sekar Selvaraj Vasu Vel Velan Vetri
Mohammed Mohammad Ahmed Ahmad Imran Irfan Ayaz Aftab Akhtar Anwar Arif Asif
Ayub Azhar Farhan Faisal Farooq Ghulam Hamid Hanif Iqbal Ismail Jaffar Kamran
Khalid Naveed Nadeem Nasir Parvez Qasim Rafiq Rashid Riyaz Rizwan Sadiq
Saleem Salman Shabbir Shahid Shakeel Shoaib Tariq Waseem Yasin Yusuf Zafar
Zahid Zakir Zia Aslam Bilal Danish Faizan Hamza Junaid Kabir Rehan Sohail
Tanveer Umar Wasim Aamir Adil Ali Anas Arshad Basit Ehsan Fahad Ghalib Habib
Ikram Javed Kashif Liaqat Mansoor Nawaz Owais Pervez Qadir Rafi
Gurpreet Harpreet Jaspreet Manpreet Amrit Amritpal Balwinder Baljeet
Baljinder Balraj Bhupinder Charanjit Daljeet Daljit Davinder Gurbaksh
Gurcharan Gurdeep Gurdev Gurinder Gurjeet Gurjit Gurmeet Gurmukh Gursewak
Gurtej Harbans Harbhajan Hardeep Hardev Harinder Harjeet Harjit Harjot
Harminder Harpal Harvinder Inderjit Jagdeep Jagjit Jagmohan Jagtar Jarnail
Jaswant Joginder Kuldeep Kulwinder Lakhwinder Mahinder Mandeep Manjeet
Manjit Manmohan Narinder Navdeep Paramjit Parminder Pritpal Rajinder Ranjit
Sarabjit Satnam Sukhbir Sukhdev Sukhjinder Sukhwinder Sukhpal Surinder
Tarsem Tejinder Waryam
Abhijit Amitava Anirban Anupam Arindam Arka Asit Bappa Bibhas Bikash Biman
Biplab Chandan Debashish Debu Dilip Gautam Indranil Jayanta Kajal Kalyan
Kartick Kaushik Koushik Malay Manas Manik Mrinal Nabin Naren Nirmal Nitai
Palash Partha Pranab Prasenjit Provash Rabindra Rana Ranadeb Ranjan Ratan
Rituparno Sagar Sandip Sanjib Santanu Satyajit Shiva Shouvik Somenath
Somnath Subrata Sujit Sukanta Sumit Suprakash Suvo Swagata Swapan Tamal
Tanmoy Tapan Utpal
Alpesh Ashwin Bhavesh Chirag Dhaval Jayesh Jignesh Kalpesh Kamlesh Ketan
Kishor Mayur Mehul Mihir Nayan Nilesh Nishith Pratik Rajendra Tejas Umang
Viral
Abhijeet Amol Ashutosh Avinash Dattatray Hrishikesh Nilesh Omkar Pravin
Sagar Sandeep Santosh Shashikant Shrikant Suhas Swapnil Vaibhav Vinayak
Vishwas
""".split()

FEMALE_NAMES = """
Priya Neha Pooja Kavita Anita Sunita Sangeeta Anjali Deepika Shweta Ritu
Rekha Meena Meera Geeta Sita Radha Lakshmi Saraswati Parvati Durga Kiran
Kirti Nisha Rashmi Jyoti Suman Usha Uma Vandana Varsha Vidya Vimla Yamini
Alka Aarti Asha Bharti Chanda Chandni Damini Divya Ekta Gauri Geetanjali
Hema Indira Ishita Kajal Kalpana Kamala Kanchan Karuna Komal Lata Madhuri
Mala Malti Mamta Manisha Manju Mayuri Mohini Mridula Namita Nandini Nayana
Nidhi Nikita Nirmala Pallavi Pratibha Preeti Priyanka Rachna Rajni Rani
Ranjana Reena Renu Roopa Ruchi Sandhya Sapna Sarika Savita Seema Shalini
Shanta Shanti Sheela Shilpa Shobha Shobhana Shraddha Simran Smita Snehal
Sonal Sonam Sonia Sudha Sujata Sulochana Sumitra Supriya Swati Tanuja Tanvi
Tara Urmila Vanita Veena Vibha Vinita Yogita
Meenakshi Kaveri Kamakshi Parvathi Padma Saroja Vasantha Yamuna Sarasu
Janaki Valli Bhuvana Bhuvaneshwari Chitra Deepa Devika Gayathri Girija Gowri
Jayalakshmi Kalyani Kanaka Kasturi Kavya Lalitha Latha Leela Lokeshwari
Madhavi Malathi Malliga Malar Mangala Meenal Muthulakshmi Nagalakshmi Nalini
Padmavathi Parimala Pushpa Rajeshwari Rajini Ramya Rangammal Revathi Saranya
Selvi Shanthi Sharadha Shobana Sivagami Soumya Subbulakshmi Suguna Sujatha
Sumathi Swarnalatha Tamilarasi Thara Thulasi Vaidehi Vani Vasanthi Vennila
Vijaya Vimala Viji
Fatima Ayesha Zainab Sana Sania Farah Farida Fauzia Gulnaz Humaira Iram
Kausar Khadija Mahira Maryam Mehjabeen Mumtaz Nadia Naseem Nazia Nazneen
Noor Nusrat Parveen Raheela Rashida Razia Rehana Rubina Rukhsana Rukhsar
Rushda Saba Sabiha Sadia Safia Sajida Salma Samina Shabana Shabnam Shahnaz
Shaheen Shakila Shamim Sharmeen Shazia Sofia Sumaira Tahira Tasneem Uzma
Yasmeen Zeba Zeenat Zubeida Amina Asma Bushra Farzana Hina Ismat Lubna
Amarjit Jaswinder Kamaljit Kiranjit Lakhbir Lovepreet Navjot Pritam
Rajwinder Ravneet Sarbjit Satwinder Simarjit
Anima Anindita Ananya Aparna Arpita Bandana Bandita Bharati Chaitali
Chandana Chumki Debashree Doyel Gargi Ipsita Jaya Jhumur Kabita Kakoli
Kanika Keya Labani Madhabi Mahua Malabika Mallika Mausumi Mimi Mita Mitali
Moloya Mousumi Nabanita Papiya Parama Paromita Piyali Pritha Purabi Purnima
Rakhi Rinku Rupa Rupali Sanchari Sangita Sarama Sathi Shabari Shampa
Sharmila Sipra Soma Srabani Suchitra Sudeshna Sunanda Swagata Swarnali
Tanushree Titli Tripti Tumpa
Alpa Bhavna Bijal Darshana Dhara Falguni Hansa Hetal Jigna Krupa Mira Palak
Priti Rachana Sejal Toral Urvi Vidhi
Ashwini Bhagyashree Deepali Ketaki Manasi Manjiri Manjusha Meghana Mrunal
Prajakta Radhika Rajashree Rutuja Shalaka Shubhangi Snehal Suchita Sushma
Swapnali Trupti Ujwala Vaishali
""".split()


def build_dataset():
    male = {n.lower() for n in MALE_NAMES}
    female = {n.lower() for n in FEMALE_NAMES}
    ambiguous = male & female
    male -= ambiguous
    female -= ambiguous

    rows = [(n, "M") for n in sorted(male)] + [(n, "F") for n in sorted(female)]

    out_path = pathlib.Path(__file__).parent.parent / "data" / "names.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "gender"])
        writer.writerows(rows)

    print(f"wrote {out_path}: {len(male)} male, {len(female)} female, "
          f"{len(ambiguous)} dropped as ambiguous ({sorted(ambiguous)})")
    return out_path


if __name__ == "__main__":
    build_dataset()
