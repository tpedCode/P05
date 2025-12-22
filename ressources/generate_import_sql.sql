-- Table olist_customers_dataset
CREATE TABLE olist_customers_dataset (
  customer_id VARCHAR(255),
  customer_unique_id VARCHAR(255),
  customer_zip_code_prefix INT,
  customer_city VARCHAR(255),
  customer_state VARCHAR(255)
);
LOAD DATA LOCAL INFILE 'C:\\Users\\barre\\Documents\\Pro\\Reconversion professionnelle\\Formations\\Data Scientist by Openclassrooms\\P05\\data\\olist_customers_dataset.csv'
INTO TABLE olist_customers_dataset
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


-- Table olist_geolocation_dataset
CREATE TABLE olist_geolocation_dataset (
  geolocation_zip_code_prefix INT PRIMARY KEY,
  geolocation_lat FLOAT,
  geolocation_lng FLOAT,
  geolocation_city VARCHAR(255),
  geolocation_state VARCHAR(255)
);
LOAD DATA LOCAL INFILE 'C:\\Users\\barre\\Documents\\Pro\\Reconversion professionnelle\\Formations\\Data Scientist by Openclassrooms\\P05\\data\\olist_geolocation_dataset.csv'
INTO TABLE olist_geolocation_dataset
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


-- Table olist_orders_dataset
CREATE TABLE olist_orders_dataset (
  order_id VARCHAR(255),
  customer_id VARCHAR(255),
  order_status VARCHAR(255),
  order_purchase_timestamp VARCHAR(255),
  order_approved_at VARCHAR(255),
  order_delivered_carrier_date VARCHAR(255),
  order_delivered_customer_date VARCHAR(255),
  order_estimated_delivery_date VARCHAR(255)
);
LOAD DATA LOCAL INFILE 'C:\\Users\\barre\\Documents\\Pro\\Reconversion professionnelle\\Formations\\Data Scientist by Openclassrooms\\P05\\data\\olist_orders_dataset.csv'
INTO TABLE olist_orders_dataset
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


-- Table olist_order_items_dataset
CREATE TABLE olist_order_items_dataset (
  order_id VARCHAR(255),
  order_item_id INT,
  product_id VARCHAR(255),
  seller_id VARCHAR(255),
  shipping_limit_date VARCHAR(255),
  price FLOAT,
  freight_value FLOAT
);
LOAD DATA LOCAL INFILE 'C:\\Users\\barre\\Documents\\Pro\\Reconversion professionnelle\\Formations\\Data Scientist by Openclassrooms\\P05\\data\\olist_order_items_dataset.csv'
INTO TABLE olist_order_items_dataset
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


-- Table olist_order_payments_dataset
CREATE TABLE olist_order_payments_dataset (
  order_id VARCHAR(255),
  payment_sequential INT,
  payment_type VARCHAR(255),
  payment_installments INT,
  payment_value FLOAT
);
LOAD DATA LOCAL INFILE 'C:\\Users\\barre\\Documents\\Pro\\Reconversion professionnelle\\Formations\\Data Scientist by Openclassrooms\\P05\\data\\olist_order_payments_dataset.csv'
INTO TABLE olist_order_payments_dataset
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


-- Table olist_order_reviews_dataset
CREATE TABLE olist_order_reviews_dataset (
  review_id VARCHAR(255),
  order_id VARCHAR(255),
  review_score INT,
  review_comment_title VARCHAR(255),
  review_comment_message VARCHAR(255),
  review_creation_date VARCHAR(255),
  review_answer_timestamp VARCHAR(255)
);
LOAD DATA LOCAL INFILE 'C:\\Users\\barre\\Documents\\Pro\\Reconversion professionnelle\\Formations\\Data Scientist by Openclassrooms\\P05\\data\\olist_order_reviews_dataset.csv'
INTO TABLE olist_order_reviews_dataset
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


-- Table olist_products_dataset
CREATE TABLE olist_products_dataset (
  product_id VARCHAR(255),
  product_category_name VARCHAR(255),
  product_name_lenght FLOAT,
  product_description_lenght FLOAT,
  product_photos_qty FLOAT,
  product_weight_g FLOAT,
  product_length_cm FLOAT,
  product_height_cm FLOAT,
  product_width_cm FLOAT
);
LOAD DATA LOCAL INFILE 'C:\\Users\\barre\\Documents\\Pro\\Reconversion professionnelle\\Formations\\Data Scientist by Openclassrooms\\P05\\data\\olist_products_dataset.csv'
INTO TABLE olist_products_dataset
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


-- Table olist_sellers_dataset
CREATE TABLE olist_sellers_dataset (
  seller_id VARCHAR(255),
  seller_zip_code_prefix INT,
  seller_city VARCHAR(255),
  seller_state VARCHAR(255)
);
LOAD DATA LOCAL INFILE 'C:\\Users\\barre\\Documents\\Pro\\Reconversion professionnelle\\Formations\\Data Scientist by Openclassrooms\\P05\\data\\olist_sellers_dataset.csv'
INTO TABLE olist_sellers_dataset
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


-- Table product_category_name_translation
CREATE TABLE product_category_name_translation (
  product_category_name VARCHAR(255),
  product_category_name_english VARCHAR(255)
);
LOAD DATA LOCAL INFILE 'C:\\Users\\barre\\Documents\\Pro\\Reconversion professionnelle\\Formations\\Data Scientist by Openclassrooms\\P05\\data\\product_category_name_translation.csv'
INTO TABLE product_category_name_translation
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
