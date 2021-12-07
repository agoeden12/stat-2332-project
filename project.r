# Question 1:
d1 <- read.csv("final.csv")

# Question 2:
nrow(d1)
ncol(d1)

# Question 3:
str(d1)

# Question 4:
d1 <- subset(d1, select = -c(ID))
str(d1)

# Question 5:
sapply(d1[c("MOFB", "YOB", "AOR")], function(x) sum(is.na(x)))

# Question 6:
d2 <- d1[c("RMOB", "WI", "RCA", "Religion", "Region", "AOR", "HEL", "DOBCMC", "DOFBCMC", "MTFBI", "RW", "RH", "RBMI")]
str(d2)

# Question 7:
d3 <- na.omit(d2)
str(d3)

# Question 8:
summary(d3)

# Question 9:
d3$AVG <- rowMeans(d3[,c("DOBCMC", "DOFBCMC", "MTFBI")])

# Question 10:
d3$Newreligion[d3$Religion == 1] <- 1
d3$Newreligion[d3$Religion != 1] <- 2

# Question 11:
table(d3$Region)

# Question 12:
table(d3$Region, d3$Religion)

# Question 13:
aggregate(AOR~Region, data = d3, mean)

# Question 14:
aggregate(AOR~Religion, data = d3, var)

#---- Needed for 15-18
par(mfrow = c(2, 2))

# Question 15:
labels <- factor(d3$MTFBI)
boxplot(table(labels), main = "Boxplot of MTFBI")

# Question 16:
labels <- factor(d3$RCA)
hist(table(labels), main = "Histogram of RCA")

# Question 17:
labels <- factor(d3$Region)
barplot(table(labels), main = "Bar graph of Region")

# Question 18:
labels <- factor(d3$Region)
pie(table(labels), main = "Pie chart of Region")

# Question 19:
print("-- Graphs displayed --")

# Question 20:
d4 <- with(d3, split(d3, WI))
length(d4)

# Question 21:
result <- matrix(NA, length(d4), 5)
colnames(result) <- c("Mean", "Min", "Max", "Var", "STD")
for(i in 1:length(d4)) { # nolint
    group <- d4[[i]]
    mean <- round(mean(group$MTFBI), 1)
    min <- round(min(group$MTFBI), 1)
    max <- round(max(group$MTFBI), 1)
    var <- round(var(group$MTFBI), 1)
    med <- round(median(group$MTFBI), 1)
    result[i, ] <- c(mean, min, max, var, med)
}
result

# Question 22:
t.test(d3$MTFBI, mu = 30)

# Question 23:
shapiro.test(d3[1:5000, "MTFBI"])

# Question 24:
t.test(d3$MTFBI~d3$Newreligion)

# Question 25:
cor(d3[, c("DOBCMC", "DOFBCMC", "AOR", "MTFBI", "RW", "RH", "RBMI")])

# Question 26:
print("-- No Question --")

# Question 27:
summary(lm(MTFBI~AOR + RW + Religion, data = d3))

# Question 28:
true_mean <- 640
true_variance <- 257920

simdata <- function(n) {

    x <- rbinom(n, 20, .7)
    u <- runif(n, min = 15, max = 30)
    n1 <- rnorm(n, 0, 5)
    e <- runif(n, min = -1, max = 1)

    y <- 50 + (10 * x) + (20 * u) + (100 * n1) + e
    y
}
simdata(100)


# Question 29 & 30:

result_function <- function(n) {
    x <- simdata(n)
    mean.x <- mean(x)
    var.x <- var(x)
    mv <- c(mean.x, var.x)
    mv
}

a <- replicate(100, result_function(1000))

sim_mean <- mean(a[1, ])
sim_var <- var(a[2, ])

abs(sim_mean - true_mean)
abs(sim_var - true_variance)

# Question 31 & 32:
b <- replicate(500, result_function(1000))

sim_mean <- mean(b[1, ])
sim_var <- var(b[2, ])

abs(sim_mean - true_mean)
abs(sim_var - true_variance)

# Question 33:
for (x in 1:5) {
    y <- x + 1
    z <- y + 1

    result <-  (exp(1)^x - log10(z^z)) / (5 + y)
    print(result)
}
 
# Question 34:
x <- matrix(c(70, 100, 40,
              120, 450, 340,
              230, 230, 1230), 3, 3, byrow = TRUE)
y <- c(900, 1000, 1230)
sol <- solve(x, y)
print(sol)

# Question 35:
A <- matrix(c(20, 30, 30,
              20, 80, 120,
              40, 90, 360), 3, 3, byrow = TRUE)
solve(A)

# Question 36:
b <- c(10, 20, 30)
solve(t(A) * A) * t(A) * b

# Question 37:
curve((exp(1)^x) / factorial(x), from = 2, to = 15)

# # Question 38:
# continuous <- function(x) {
#     if (x < 0) {
#         (2 * (x^2)) + exp(1)^x + 3
#     } else if (x >= 0 || x < 10) {
#         (9 * x) + log10(20)
#     } else {
#         (7 * (x^2)) + (5 * x) - 17
#     }
# }

# x <- -1000:1000
# y <- replicate(length(x), continuous(x))
# plot(x, y, type = "l")

# Question 39:
radii <- 10:19
print(pi * radii^2)

# Question 40:
x <- 2:10000
fx <- 1 / log10(x)
print(sum(fx))

# Question 41:
result <- 0
for (i in 1:30) {
    for (j in 1:10) {
        result <- result + ((i^10) / (3 + j))
    }
}
print(result)

# Question 42:
integrate(function(x) (x^15 * exp(1)^(-40 * x)), lower = 0, upper = Inf)

# Question 43:
integrate(function(x) (x^150 * (1 - x)^30), lower = 0, upper = 1)

# Question 44:
for (x in 1:5) {
    y <- x + 1
    z <- y + 1

    result <-  (exp(1)^x - log10(z^z)) / (5 + y)
    print(result)
}

# Question 45:
quadratic <- function(a, b, c) {
  result <- c((-b + sqrt(b^2 - 4 * a * c)) / (2 * a),
              (-b - sqrt(b^2 - 4 * a * c)) / (2 * a))
  result
}
print(quadratic(1, -33, 1))

# Question 46:
print("-- No Question --")

# Question 47:
p <- 40
t <- 50
r <- 0.10
print(p * (1 + r)^t)