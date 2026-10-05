SELECT
    c.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    c.tenure_months,

    s.phone_service,
    s.multiple_lines,
    s.internet_service,
    s.online_security,
    s.online_backup,
    s.device_protection,
    s.tech_support,
    s.streaming_tv,
    s.streaming_movies,

    b.contract_type,
    b.paperless_billing,
    b.payment_method,
    b.monthly_charges,
    b.total_charges,

    c.churn

FROM customers c

JOIN customer_services s
    ON c.customer_id = s.customer_id

JOIN customer_billing b
    ON c.customer_id = b.customer_id;