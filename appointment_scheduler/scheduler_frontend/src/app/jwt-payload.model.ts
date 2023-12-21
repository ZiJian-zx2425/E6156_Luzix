// src/app/jwt-payload.model.ts

export interface JwtPayload {
    sub: string;
    name: string;
    role: string;
    // You can also include standard JWT fields like exp (expiration), iat (issued at), etc.
    exp?: number;
}
