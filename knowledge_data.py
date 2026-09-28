# mwg_chatbot_python_service/knowledge_data.py
# MWG AI WhatsApp Sales Agent — Knowledge Base (Chapter 10)
#
# Single source of truth for the RAG knowledge chunks. Edit this file and run:
#     py -3.10 seed_knowledge.py            (re-embeds + replaces the collection)
#     py -3.10 seed_knowledge.py --dry-run  (shows what would change)
#
# Rules for writing chunks:
#   - NO car/bike service prices here — those come live from the app's
#     `services` / `coupons` collections on every message.
#   - NO guaranteed income / ROI / profit numbers (sales rule #1).
#   - NO BYOB (discontinued) — only the one redirect chunk below.
#   - Contact number is ONLY +91 94296 91299.
#
# Categories (must match INTENT_CATEGORY in main.py):
#   franchise | area_partner | services | job | general_faq

PHONE = "+91 94296 91299"
WEBSITE = "www.mrwhitegloves.com"

KNOWLEDGE = [
    # ───────────────────────────── FRANCHISE ─────────────────────────────
    {
        "id": "fr_001", "category": "franchise",
        "text": "Q: What is MWG franchise and how does it work?\n"
                "A: Mr. White Gloves Franchise is a doorstep car & bike wash business. You invest once, MWG gives you "
                "complete self-powered car wash kits, training, hiring support, business setup guidance and marketing "
                "support. Your team washes cars at the customer's doorstep in your area/city and you earn 38% commission "
                "on every service. There are 3 plans: Solo Partner, Growth Partner and Master Franchise.",
    },
    {
        "id": "fr_002", "category": "franchise",
        "text": "Q: How much does MWG franchise cost? What is the franchise price?\n"
                "A: MWG has 3 franchise plans (all prices + 18% GST): "
                "(1) Solo Partner — ₹99,000 + GST (about ₹1,16,820 total). "
                "(2) Growth Partner — ₹1,85,000 + GST (about ₹2,18,300 total) — most popular. "
                "(3) Master Franchise — ₹5,00,000 + GST (about ₹5,90,000 total).",
    },
    {
        "id": "fr_003", "category": "franchise",
        "text": "Q: What do I get in the Solo Partner plan?\n"
                "A: Solo Partner (₹99,000 + GST): 1 kit (1 washer), 1 technician, coverage of 1 zone/area, capacity of "
                "8-12 cars per day, break-even usually in 3-6 months, 38% commission on every service. No sub-franchise "
                "rights. Best for first-time business owners, homemakers and retired professionals.",
    },
    {
        "id": "fr_004", "category": "franchise",
        "text": "Q: What do I get in the Growth Partner plan?\n"
                "A: Growth Partner (₹1,85,000 + GST) — our most popular plan: 2 kits (2 washers), 2-3 technicians, "
                "coverage of 2-3 zones, capacity of 20-30 cars per day, break-even usually in 4-8 months, 38% commission "
                "on every service. No sub-franchise rights. Best for small business owners, ex-corporate professionals "
                "and side-business builders.",
    },
    {
        "id": "fr_005", "category": "franchise",
        "text": "Q: What do I get in the Master Franchise plan? Can I appoint sub-franchises?\n"
                "A: Master Franchise (₹5,00,000 + GST): 4 kits, 5-8 technicians, coverage of an entire city/district, "
                "capacity of 50-80 cars per day, break-even usually in 6-12 months, 38% commission on every service. "
                "It is the ONLY plan with sub-franchise rights — you can appoint sub-partners in your city. Best for "
                "investors who want city exclusivity.",
    },
    {
        "id": "fr_006", "category": "franchise",
        "text": "Q: Which franchise plan is right for me?\n"
                "A: Choose by budget and how big you want to start: Solo Partner (₹99,000 + GST) to start small with 1 "
                "kit in 1 area; Growth Partner (₹1,85,000 + GST) to cover 2-3 zones with 2 kits; Master Franchise "
                "(₹5,00,000 + GST) for a whole city with sub-franchise rights. You can start small and grow later.",
    },
    {
        "id": "fr_007", "category": "franchise",
        "text": "Q: How much can I earn from MWG franchise? What is the ROI or profit?\n"
                "A: You earn 38% commission on every service your team completes. Typical break-even: Solo Partner 3-6 "
                "months, Growth Partner 4-8 months, Master Franchise 6-12 months. Actual earnings depend on your city, "
                "team size and how many cars you serve — MWG does not promise a fixed monthly income. Our team can walk "
                "you through numbers for your city on a call.",
    },
    {
        "id": "fr_008", "category": "franchise",
        "text": "Q: What is included in the MWG franchise kit? What equipment do I get?\n"
                "A: Each kit is a complete self-powered doorstep car wash setup with 25+ premium items — equipment, "
                "chemicals, tools and branding — such as pressure washer, vacuum cleaner, polish machine, foam cannon, "
                "detailing kit, microfibre cloths, power backup and branded uniforms. Solo Partner gets 1 kit, Growth "
                "Partner 2 kits and Master Franchise 4 kits.",
    },
    {
        "id": "fr_009", "category": "franchise",
        "text": "Q: What support does MWG provide to franchise partners?\n"
                "A: MWG supports you end to end: (1) complete car wash kits, (2) 5-day training, (3) technician hiring "
                "and on-field training support, (4) business setup and launch guidance, (5) online and offline marketing "
                "with customer leads and app bookings, (6) MWG brand and branding material.",
    },
    {
        "id": "fr_010", "category": "franchise",
        "text": "Q: Do I need an office space or shop for MWG franchise?\n"
                "A: No office or shop is required. The MWG franchise is fully mobile — kits can be stored at your home or "
                "any convenient place, and technicians travel to the customer's location for every wash.",
    },
    {
        "id": "fr_011", "category": "franchise",
        "text": "Q: Do I need to find customers myself for the franchise?\n"
                "A: MWG runs online marketing and brings customer leads and bookings through the MWG app and website. You "
                "also get marketing material for local promotion. Local effort (societies, offices, word-of-mouth) helps "
                "you grow faster.",
    },
    {
        "id": "fr_012", "category": "franchise",
        "text": "Q: What is the step by step process to get MWG franchise? How to apply?\n"
                "A: Franchise onboarding: (1) Discussion call — your questions and a demand check for your city. "
                "(2) Choose your plan and pay a token amount to lock your city/area. (3) Complete the payment and sign "
                "the agreement. (4) MWG team helps you set up — kits, training, staff and launch marketing. "
                f"To start, call or WhatsApp {PHONE}.",
    },
    {
        "id": "fr_013", "category": "franchise",
        "text": "Q: Who handles employee hiring and training for the franchise?\n"
                "A: MWG helps you hire technicians through online job portals and trains them on the field. Ready-made "
                "training videos are also available for your staff anytime.",
    },
    {
        "id": "fr_014", "category": "franchise",
        "text": "Q: Do I have to manage the MWG franchise business myself?\n"
                "A: It is your choice. Managing it yourself gives the best margin, but you can also run it with hired "
                "staff or a manager once the business is established.",
    },
    {
        "id": "fr_015", "category": "franchise",
        "text": "Q: Is MWG franchise available in my city?\n"
                "A: Franchise territories are given city-wise and zone-wise. Share your city (and pincode) and our team "
                "will check if your city or zone is still open and help you lock it before someone else does.",
    },
    {
        "id": "fr_016", "category": "franchise",
        "text": "Q: Is GST included in the franchise price?\n"
                "A: No. All franchise prices are plus 18% GST: Solo Partner ₹99,000 + GST (about ₹1,16,820), Growth "
                "Partner ₹1,85,000 + GST (about ₹2,18,300), Master Franchise ₹5,00,000 + GST (about ₹5,90,000).",
    },
    {
        "id": "fr_017", "category": "franchise",
        "text": "Q: Why should I choose MWG franchise?\n"
                "A: MWG solves real problems for car owners — no need to go to a car wash (we come to the doorstep), "
                "consistent premium quality, and eco-friendly waterless methods. As a partner you get a known brand, the "
                "MWG app for bookings, complete kits, training and marketing support, and plans starting at ₹99,000 + GST.",
    },
    {
        "id": "fr_018", "category": "franchise",
        "text": "Q: What is BYOB (Be Your Own Boss)? Is BYOB still available?\n"
                "A: The BYOB model has been discontinued and is no longer available. Today you can join MWG as an Area "
                "Partner (₹15,000 + GST) or as a Franchise partner (Solo Partner, Growth Partner or Master Franchise).",
    },

    # ──────────────────────────── AREA PARTNER ───────────────────────────
    {
        "id": "ap_001", "category": "area_partner",
        "text": "Q: What is an Area Partner in Mr. White Gloves?\n"
                "A: An Area Partner is a local business owner who delivers MWG car wash services in a fixed area — "
                "societies and residential complexes. It is not a job, it is your own business backed by the MWG brand. "
                "Jitne zyada subscribers, utni zyada income.",
    },
    {
        "id": "ap_002", "category": "area_partner",
        "text": "Q: How much investment is needed to become an Area Partner?\n"
                "A: The Area Partner investment is ₹15,000 + 18% GST (about ₹17,700 total). It is the most affordable "
                "way to start your own business with MWG.",
    },
    {
        "id": "ap_003", "category": "area_partner",
        "text": "Q: How does the Area Partner model work?\n"
                "A: Area Partners work on monthly subscriptions in a fixed locality (societies and residential "
                "complexes). Customers subscribe monthly for regular car cleaning, the MWG app handles online bookings, "
                "and you focus on service delivery. More subscribers = more stable monthly income.",
    },
    {
        "id": "ap_004", "category": "area_partner",
        "text": "Q: Is Area Partner a job or a business?\n"
                "A: Area Partner is NOT a job — it is your own business. You own your area and customer relationships "
                "and are not an employee of MWG. Your income grows with your effort and subscriber count.",
    },
    {
        "id": "ap_005", "category": "area_partner",
        "text": "Q: How does customer acquisition work for Area Partners?\n"
                "A: It is a team effort. MWG generates demand online — digital marketing, app and website enquiries, "
                "social media, society demos and events. The Area Partner converts locally — presence in societies, "
                "relationships, word-of-mouth, trial wash demos and great service that turns into subscriptions.",
    },
    {
        "id": "ap_006", "category": "area_partner",
        "text": "Q: What are the growth phases for an Area Partner?\n"
                "A: Phase 1 — Operator mode: you personally do the service, starting with 20-40 cars to learn the "
                "business and build trust. Phase 2 — Growth mode: once established, hire helpers and move to supervision "
                "while keeping quality high. First master the craft, then build the business.",
    },
    {
        "id": "ap_007", "category": "area_partner",
        "text": "Q: What support does MWG give to Area Partners?\n"
                "A: (1) Brand & identity — uniform and ID card. (2) Starter kit with essential equipment. (3) Training & "
                "SOPs for quality standards. (4) MWG app access for customer management. (5) Online and offline "
                "marketing support. (6) Quality monitoring and growth support.",
    },
    {
        "id": "ap_008", "category": "area_partner",
        "text": "Q: Who can become an Area Partner? What are the requirements?\n"
                "A: Anyone ready for hard work — delivery riders, workers looking for stable income, students wanting "
                "part-time business, anyone who wants to grow. Requirements: Aadhaar card, a smartphone with WhatsApp, "
                "the ₹15,000 + GST investment, and commitment to start within 7 days.",
    },
    {
        "id": "ap_009", "category": "area_partner",
        "text": "Q: How much can an Area Partner earn?\n"
                "A: Income depends on how many monthly subscribers you serve in your area — the more cars, the more "
                "monthly income. MWG does not promise a fixed income; our team can explain the numbers for your area.",
    },
    {
        "id": "ap_010", "category": "area_partner",
        "text": "Q: How to become an Area Partner? How to join?\n"
                f"A: Call or WhatsApp {PHONE} or visit {WEBSITE}. Slots are limited per area and given on a "
                "first-come, first-served basis. Share your city and area so we can check if your locality is open.",
    },
    {
        "id": "ap_011", "category": "area_partner",
        "text": "Q: What is the difference between Area Partner and Franchise?\n"
                "A: Area Partner (₹15,000 + GST): you personally serve a fixed locality/societies on monthly "
                "subscriptions. Franchise (₹99,000 to ₹5,00,000 + GST): you run a team with 1-4 kits covering zones or "
                "a whole city, with full setup, training and marketing support.",
    },
    {
        "id": "ap_012", "category": "area_partner",
        "text": "Q: What is the daily routine of an Area Partner?\n"
                "A: A typical day starts around 6:30 AM with car cleaning rounds in societies — 2-3 hours to clean your "
                "subscribers' cars, then WhatsApp updates to customers and record keeping. Most work is done by late "
                "morning, leaving the rest of the day free.",
    },

    # ────────────────────────────── SERVICES ─────────────────────────────
    # (No prices here — live price list comes from the app on every message)
    {
        "id": "sv_001", "category": "services",
        "text": "Q: Is the car wash done at my home or do I need to go somewhere?\n"
                "A: All Mr. White Gloves services are 100% doorstep. Our trained technician comes to your home, office "
                "or any location with all equipment. You do not need to go anywhere.",
    },
    {
        "id": "sv_002", "category": "services",
        "text": "Q: What car and bike wash services does MWG offer?\n"
                "A: Car: Complete Exterior Body Wash, Basic Wash, Standard Wash, Premium Wash and a Monthly "
                "Subscription. Bike: Bike Wash, Wash & Polishing and Bike Detailing (plus a bike monthly plan). Prices "
                "depend on the vehicle type (Hatchback, Sedan, SUV or Two wheeler) — share your vehicle and location "
                "to get the exact price.",
    },
    {
        "id": "sv_003", "category": "services",
        "text": "Q: How do I book a car or bike wash?\n"
                "A: Book in the MWG app (search \"Mr White Gloves\" on Google Play or the App Store), on "
                f"{WEBSITE}, or call {PHONE}. Apply any active coupon code in the app while booking.",
    },
    {
        "id": "sv_004", "category": "services",
        "text": "Q: Do I need to be present during the car wash?\n"
                "A: No, you do not need to be present. Your car just needs to be accessible at the booked location — "
                "our trained staff handles everything.",
    },
    {
        "id": "sv_005", "category": "services",
        "text": "Q: Is MWG service safe for all car types?\n"
                "A: Yes. MWG services are safe for hatchbacks, sedans, SUVs, luxury cars and two-wheelers. Our trained "
                "staff uses the right equipment and products for each vehicle.",
    },
    {
        "id": "sv_006", "category": "services",
        "text": "Q: Is the car wash eco friendly?\n"
                "A: Yes. MWG uses eco-friendly products and water-efficient, waterless cleaning methods, avoiding harsh "
                "chemicals that damage paint or harm the environment.",
    },
    {
        "id": "sv_007", "category": "services",
        "text": "Q: What payment methods are accepted?\n"
                "A: You can pay online in the MWG app — UPI (GPay, PhonePe, Paytm), debit/credit cards and net banking. "
                f"For payment questions call {PHONE}.",
    },
    {
        "id": "sv_008", "category": "services",
        "text": "Q: Can I reschedule or cancel my booking?\n"
                f"A: Yes. You can reschedule or cancel from the MWG app, or call/WhatsApp {PHONE} before your slot.",
    },
    {
        "id": "sv_009", "category": "services",
        "text": "Q: What is the monthly car wash subscription?\n"
                "A: The Monthly Subscription is a recurring plan where your car is cleaned regularly through the month "
                "at your doorstep — exterior cleaning on most days, interior cleaning on some days, plus extras like a "
                "free foam wash and car perfume. Exact inclusions and price depend on your vehicle type. To cancel a "
                f"subscription, contact support at {PHONE}.",
    },
    {
        "id": "sv_010", "category": "services",
        "text": "Q: What if I am not satisfied with the wash quality?\n"
                f"A: Customer satisfaction is our priority. If you are not happy, contact us right away at {PHONE} and "
                "we will fix it.",
    },
    {
        "id": "sv_011", "category": "services",
        "text": "Q: How long does a car or bike wash take?\n"
                "A: It depends on the package — from about 25-30 minutes for a quick exterior or bike wash up to about 3 "
                "hours for Premium Wash detailing.",
    },
    {
        "id": "sv_012", "category": "services",
        "text": "Q: Are there any offers or coupon codes?\n"
                "A: Active coupon codes are shared in the chat with the final price for your package. Apply the code in "
                "the MWG app while booking.",
    },

    # ──────────────────────────────── JOBS ───────────────────────────────
    {
        "id": "job_bda_001", "category": "job",
        "text": "Q: What is the Business Development Associate job at MWG?\n"
                "A: Mr. White Gloves is hiring a Business Development Associate for franchise expansion — identify "
                "franchise opportunities, build partner relationships, drive market expansion and support onboarding of "
                "new franchise partners. Location: Jamshedpur, Jharkhand. Full-time, work from office, 10:00 AM to "
                "7:30 PM.",
    },
    {
        "id": "job_bda_002", "category": "job",
        "text": "Q: What is the salary for Business Development Associate at MWG?\n"
                "A: Business Development Associate: ₹2.40 LPA to ₹5.80 LPA fixed, plus performance incentives. Fast "
                "career growth in a growing brand.",
    },
    {
        "id": "job_bda_003", "category": "job",
        "text": "Q: What are the responsibilities and eligibility for Business Development Associate?\n"
                "A: Responsibilities: find new franchise opportunities, build relationships with prospective partners, "
                "market research for new locations, present the franchise model, support onboarding and coordinate with "
                "marketing and operations. Eligibility: Bachelor's degree (Business/Marketing preferred, MBA a plus), "
                "1-3 years in sales/business development preferred — freshers with strong communication skills can apply.",
    },
    {
        "id": "job_bda_004", "category": "job",
        "text": "Q: How to apply for Business Development Associate job at MWG?\n"
                f"A: Send your resume to franchise@mrwhitegloves.com or call/WhatsApp {PHONE}. Location: Jamshedpur, "
                "Jharkhand. Freshers with strong communication skills are welcome.",
    },
    {
        "id": "job_fse_001", "category": "job",
        "text": "Q: What is the Field Sales Executive job at MWG?\n"
                "A: Field Sales Executives visit residential societies, talk to car owners, offer trial washes and "
                "convert them into monthly subscribers. Location: Jamshedpur (assigned local area, no long-distance "
                "travel). Good for people who enjoy field work and meeting people.",
    },
    {
        "id": "job_fse_002", "category": "job",
        "text": "Q: What is the salary for Field Sales Executive at MWG?\n"
                "A: Field Sales Executive: fixed salary ₹8,000 to ₹10,000 per month plus incentives per conversion and "
                "bonus on weekly targets.",
    },
    {
        "id": "job_fse_003", "category": "job",
        "text": "Q: What are the targets and responsibilities of a Field Sales Executive?\n"
                "A: Daily: 30-50 customer conversations in nearby societies, 10+ trial wash offers and 4-5 paid "
                "conversions, and coordinating with operations to start service. No degree or prior experience needed — "
                "good communication in Hindi or the local language and confidence matter most.",
    },
    {
        "id": "job_fse_004", "category": "job",
        "text": "Q: How to apply for Field Sales Executive job at MWG?\n"
                f"A: Call or WhatsApp {PHONE} or send your resume to franchise@mrwhitegloves.com. Location: Jamshedpur. "
                "Freshers are welcome.",
    },

    # ──────────────────────────── GENERAL FAQ ────────────────────────────
    {
        "id": "gf_001", "category": "general_faq",
        "text": "Q: What is Mr. White Gloves (MWG)?\n"
                "A: Mr. White Gloves is India's premium doorstep car & bike care brand. Trained technicians wash, polish "
                "and detail vehicles at the customer's home or office using eco-friendly methods, booked through the "
                "MWG app. MWG also offers franchise and Area Partner business opportunities.",
    },
    {
        "id": "gf_002", "category": "general_faq",
        "text": "Q: How do I contact Mr White Gloves?\n"
                f"A: Call or WhatsApp {PHONE}, or visit {WEBSITE}.",
    },
    {
        "id": "gf_003", "category": "general_faq",
        "text": "Q: Is Mr White Gloves available in my city?\n"
                "A: MWG is expanding across India. Share your city and pincode and we will check if doorstep service is "
                "available in your area right away.",
    },
]
