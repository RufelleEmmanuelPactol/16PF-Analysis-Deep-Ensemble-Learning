-- for migration purposes
create table Schools
(
    Id   int auto_increment
        primary key,
    Name varchar(255) not null
);

create table assessments
(
    Id         int auto_increment
        primary key,
    cfit       varchar(5) null,
    attr_A     int        null,
    attr_B     int        null,
    attr_C     int        null,
    attr_E     int        null,
    attr_F     int        null,
    attr_G     int        null,
    attr_H     int        null,
    attr_I     int        null,
    attr_L     int        null,
    attr_M     int        null,
    attr_N     int        null,
    attr_O     int        null,
    attr_Q1    int        null,
    attr_Q2    int        null,
    attr_Q3    int        null,
    attr_Q4    int        null,
    attr_EX    int        null,
    attr_AX    int        null,
    attr_TM    int        null,
    attr_IN    int        null,
    attr_SC    int        null,
    c_cert     int        null,
    grade      float      null,
    student_id int        null
);

create table datasetTag
(
    id   int auto_increment
        primary key,
    name varchar(256) null,
    constraint name
        unique (name)
);

create table students
(
    Id                 int auto_increment
        primary key,
    previous_school_id int               null,
    training           tinyint default 0 null,
    tagID              int               null,
    student_id         varchar(256)      null,
    course             varchar(20)       null,
    constraint student_id
        unique (student_id),
    constraint students_ibfk_1
        foreign key (previous_school_id) references Schools (Id)
);

create index previous_school
    on students (previous_school_id);

