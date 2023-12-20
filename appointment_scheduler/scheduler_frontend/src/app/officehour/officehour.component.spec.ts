import { ComponentFixture, TestBed } from '@angular/core/testing';

import { OfficehourComponent } from './officehour.component';

describe('OfficehourComponent', () => {
  let component: OfficehourComponent;
  let fixture: ComponentFixture<OfficehourComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [OfficehourComponent]
    });
    fixture = TestBed.createComponent(OfficehourComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
