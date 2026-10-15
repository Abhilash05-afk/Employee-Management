from pydantic import BaseModel, Field


class EmployeeData(BaseModel):

    Age: int = Field(
        ...,
        ge=18,
        le=70
    )

    BusinessTravel: str

    DailyRate: int = Field(
        ...,
        ge=0
    )

    Department: str

    DistanceFromHome: int = Field(
        ...,
        ge=0
    )

    Education: int = Field(
        ...,
        ge=1,
        le=5
    )

    EducationField: str

    EnvironmentSatisfaction: int = Field(
        ...,
        ge=1,
        le=4
    )

    Gender: str

    HourlyRate: int = Field(
        ...,
        ge=0,
        le=100
    )

    JobInvolvement: int = Field(
        ...,
        ge=1,
        le=4
    )

    JobLevel: int = Field(
        ...,
        ge=1,
        le=5
    )

    JobRole: str

    JobSatisfaction: int = Field(
        ...,
        ge=1,
        le=4
    )

    MaritalStatus: str

    MonthlyIncome: int = Field(
        ...,
        ge=0
    )

    MonthlyRate: int = Field(
        ...,
        ge=0
    )

    NumCompaniesWorked: int = Field(
        ...,
        ge=0
    )

    OverTime: str

    PercentSalaryHike: int = Field(
        ...,
        ge=0
    )

    PerformanceRating: int = Field(
        ...,
        ge=1,
        le=4
    )

    RelationshipSatisfaction: int = Field(
        ...,
        ge=1,
        le=4
    )

    StockOptionLevel: int = Field(
        ...,
        ge=0,
        le=3
    )

    TotalWorkingYears: int = Field(
        ...,
        ge=0
    )

    TrainingTimesLastYear: int = Field(
        ...,
        ge=0
    )

    WorkLifeBalance: int = Field(
        ...,
        ge=1,
        le=4
    )

    YearsAtCompany: int = Field(
        ...,
        ge=0
    )

    YearsInCurrentRole: int = Field(
        ...,
        ge=0
    )

    YearsSinceLastPromotion: int = Field(
        ...,
        ge=0
    )

    YearsWithCurrManager: int = Field(
        ...,
        ge=0
    )

class PredictionResponse(BaseModel):
    prediction: str
    attrition_probability: float