import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PostedofficehoursComponent } from './postedofficehours.component';

describe('PostedofficehoursComponent', () => {
  let component: PostedofficehoursComponent;
  let fixture: ComponentFixture<PostedofficehoursComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [PostedofficehoursComponent]
    });
    fixture = TestBed.createComponent(PostedofficehoursComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
